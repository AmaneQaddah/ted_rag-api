import os
import json
import time
import random
import re

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pinecone import Pinecone

from config import CHUNK_SIZE, STEP, EMBED_MODEL

load_dotenv()

CSV_PATH   = "ted_talks_en.csv"
CACHE_PATH = "cache_upserted_ids.json"

BATCH_UPSERT = 100
BATCH_EMBED  = 32

# =========================
# Metadata cleaning (one-time at ingest)
# =========================
def clean_val(val) -> str:
    s = str(val or "")
    s = re.sub(r"\{?\d+:\s*['\"]", "", s)   # remove "{0: '"
    s = re.sub(r"['\"]\}?", "", s)          # remove "'}" or '"}'
    s = re.sub(r"[\[\]'\"]", "", s)         # remove [] and quotes
    return s.strip()

def normalize_csv_list_like(val) -> str:
    # turns "['a','b']" OR "{0: 'a', 1: 'b'}" into "a, b"
    cleaned = clean_val(val)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return ", ".join(parts)

# =========================
# Chunking
# =========================
def chunk_text(text: str):
    text = (text or "").strip()
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += STEP

    return chunks

# =========================
# Cache helpers
# =========================
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_cache(ids_set):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(list(ids_set)), f)

# =========================
# Retry helpers
# =========================
def backoff_sleep(attempt: int):
    t = min(30.0, 1.0 * (2 ** (attempt - 1)))
    t = t * (0.7 + 0.6 * random.random())
    time.sleep(t)

def embed_batch_with_retry(oai: OpenAI, texts: list[str]):
    for attempt in range(1, 6):
        try:
            resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except (APITimeoutError, APIConnectionError, RateLimitError):
            backoff_sleep(attempt)
    raise RuntimeError("Embeddings failed after retries")

def pinecone_upsert_with_retry(index, vectors):
    for attempt in range(1, 6):
        try:
            index.upsert(vectors=vectors)
            return
        except Exception:
            backoff_sleep(attempt)
    raise RuntimeError("Pinecone upsert failed")

# =========================
# Main ingestion
# =========================
def main():
    oai = OpenAI(
        api_key=os.environ["LLMOD_API_KEY"],
        base_url="https://api.llmod.ai/v1",
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(
        os.environ.get("PINECONE_INDEX", "ted-rag"),
        host=os.environ["PINECONE_HOST"],
    )

    upserted = load_cache()

    df = pd.read_csv(CSV_PATH, encoding="utf-8", keep_default_na=False)
    print(f"Starting ingestion of {len(df)} talks...")

    pending_records = []
    vectors_buffer  = []
    ids_buffer      = []

    for _, row in df.iterrows():
        talk_id = str(row.get("talk_id", "")).strip()
        if not talk_id:
            continue

        description = str(row.get("description", "")).strip()

        # ===== extract year only from published_date (DD/MM/YYYY) =====
        pub_date = str(row.get("published_date", "")).strip()
        published_year = ""
        if re.match(r"^\d{2}/\d{2}/\d{4}$", pub_date):
            published_year = pub_date[-4:]
        else:
            m = re.search(r"(\d{4})", pub_date)
            published_year = m.group(1) if m else ""

        # ===== minimal metadata, cleaned once here =====
        meta_base = {
            "talk_id": talk_id,
            "title": str(row.get("title", "")),
            "all_speakers": normalize_csv_list_like(row.get("all_speakers", "")),
            "topics": normalize_csv_list_like(row.get("topics", "")),
            "published_year": published_year,
            "available_lang": normalize_csv_list_like(row.get("available_lang", "")),
        }

        transcript = str(row.get("transcript", ""))
        chunks = chunk_text(transcript)

        for ci, chunk in enumerate(chunks):
            chunk_id = f"{talk_id}_c{ci:03d}"
            if chunk_id in upserted:
                continue

            # ===== enrichment for better retrieval =====
            embedding_text = (
                f"Title: {meta_base['title']}\n"
                f"Speakers: {meta_base['all_speakers']}\n"
                f"Topics: {meta_base['topics']}\n"
                f"Year: {meta_base['published_year']}\n"
                f"Languages: {meta_base['available_lang']}\n"
                f"Description: {description}\n"
                f"Transcript: {chunk}"
            )

            meta = meta_base.copy()
            meta.update({
                "chunk_id": chunk_id,
                "chunk": chunk,
            })

            pending_records.append((chunk_id, embedding_text, meta))

            if len(pending_records) >= BATCH_EMBED:
                texts = [r[1] for r in pending_records]
                embs = embed_batch_with_retry(oai, texts)

                for (cid, _, m), emb in zip(pending_records, embs):
                    vectors_buffer.append({
                        "id": cid,
                        "values": emb,
                        "metadata": m,
                    })
                    ids_buffer.append(cid)

                pending_records.clear()

                if len(vectors_buffer) >= BATCH_UPSERT:
                    pinecone_upsert_with_retry(index, vectors_buffer)

                    # update cache only after successful upsert
                    for cid in ids_buffer:
                        upserted.add(cid)
                    save_cache(upserted)

                    print(f"Uploaded {len(upserted)} chunks...")
                    vectors_buffer.clear()
                    ids_buffer.clear()

    # ===== Flush leftovers =====
    if pending_records:
        texts = [r[1] for r in pending_records]
        embs = embed_batch_with_retry(oai, texts)
        for (cid, _, m), emb in zip(pending_records, embs):
            vectors_buffer.append({
                "id": cid,
                "values": emb,
                "metadata": m,
            })
            ids_buffer.append(cid)
        pending_records.clear()

    if vectors_buffer:
        pinecone_upsert_with_retry(index, vectors_buffer)
        for cid in ids_buffer:
            upserted.add(cid)
        save_cache(upserted)

    print(f"Finished! Total chunks cached: {len(upserted)}")

if __name__ == "__main__":
    main()
