import os
import re
from typing import Optional, Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

from config import (
    CHUNK_SIZE,
    OVERLAP_RATIO,
    TOP_K_QUERY,
    CHAT_MODEL,
    EMBED_MODEL,
)

app = FastAPI()

# =========================
# Schemas
# =========================
class PromptIn(BaseModel):
    question: str

# =========================
# Helpers
# =========================
def clean_val(val: Any) -> str:
    # עדיין שימושי (גם אם ניקינו ב-ingest) כי לפעמים יהיו ערכים ריקים/מוזרים
    s = str(val or "")
    s = re.sub(r"\{?\d+:\s*['\"]", "", s)
    s = re.sub(r"['\"]\}?", "", s)
    s = re.sub(r"[\[\]'\"]", "", s)
    return s.strip()

def extract_exact_n(question: str) -> Optional[int]:
    q = (question or "").lower()

    # numeric 1–3
    m = re.search(r"\b([1-3])\b", q)
    if m:
        return int(m.group(1))

    # textual variants
    if any(x in q for x in ["exactly three", "three talks", "exactly 3"]):
        return 3
    if any(x in q for x in ["exactly two", "two talks", "exactly 2"]):
        return 2

    # single talk intent
    if any(x in q for x in ["a talk", "one talk", "find a", "recommend"]):
        return 1

    return None

def extract_language_code(question: str) -> Optional[str]:
    """
    Very lightweight language detection: supports asking by code like 'he' or by common names.
    Because you saved available_lang, this enables language-filtered recommendations.
    """
    q = (question or "").lower()

    # explicit code: "in he", "language: he", "(he)"
    m = re.search(r"\b(?:in|language|lang)\s*[:=]?\s*([a-z]{2}(?:-[a-z]{2})?)\b", q)
    if m:
        return m.group(1)

    # common language names -> codes (extend if needed)
    name_map = {
        "hebrew": "he",
        "arabic": "ar",
        "english": "en",
        "french": "fr",
        "spanish": "es",
        "russian": "ru",
        "german": "de",
        "chinese": "zh-cn",
        "japanese": "ja",
        "korean": "ko",
        "portuguese": "pt",
    }
    for name, code in name_map.items():
        if name in q:
            return code

    return None

def get_clients():
    oai = OpenAI(
        api_key=os.environ["LLMOD_API_KEY"],
        base_url="https://api.llmod.ai/v1",
    )
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(
        os.environ.get("PINECONE_INDEX", "ted-rag"),
        host=os.environ["PINECONE_HOST"],
    )
    return oai, index

# =========================
# Endpoints
# =========================
@app.get("/api/stats")
def stats():
    return {
        "chunk_size": int(CHUNK_SIZE),
        "overlap_ratio": float(OVERLAP_RATIO),
        "top_k": int(TOP_K_QUERY),
    }

@app.post("/api/prompt")
async def prompt(payload: PromptIn):
    question = (payload.question or "").strip()
    oai, index = get_clients()

    # ---- Retrieval ----
    emb = oai.embeddings.create(model=EMBED_MODEL, input=question)
    q_vec = emb.data[0].embedding
    res = index.query(vector=q_vec, top_k=TOP_K_QUERY, include_metadata=True)

    # ---- Dedup by talk_id (keep best match per talk) ----
    best_by_talk: Dict[str, Dict[str, Any]] = {}
    for m in (res.matches or []):
        meta = m.metadata or {}
        tid = str(meta.get("talk_id", "")).strip()
        if not tid:
            continue
        score = float(m.score or 0)
        if tid not in best_by_talk or score > best_by_talk[tid]["score"]:
            best_by_talk[tid] = {"score": score, "meta": meta}

    sorted_talks = sorted(best_by_talk.values(), key=lambda x: x["score"], reverse=True)

    # ---- Optional language filter (only if user asked) ----
    lang_code = extract_language_code(question)
    if lang_code:
        filtered = []
        for item in sorted_talks:
            meta = item["meta"]
            langs = (meta.get("available_lang") or "")
            # available_lang is saved as "ar, bg, ...", so substring match is fine
            if re.search(rf"\b{re.escape(lang_code)}\b", str(langs)):
                filtered.append(item)
        # use filtered if it has anything; otherwise keep original (so we can still answer)
        if filtered:
            sorted_talks = filtered

    # ---- Decide how many contexts to pass to the LLM (logic improvement) ----
    n_exact = extract_exact_n(question)
    if n_exact and n_exact > 1:
        # for "exactly 2/3 talks" pass only top N distinct talks to reduce chance of wrong count
        talks_for_prompt = sorted_talks[:n_exact]
    else:
        # for single/recommend/summary/fact: pass a limited set (keeps context focused)
        talks_for_prompt = sorted_talks[: min(10, len(sorted_talks))]

    # ---- Build context ----
    context_list: List[Dict[str, Any]] = []
    context_text = ""

    for item in talks_for_prompt:
        score = item["score"]
        meta = item["meta"]

        title   = clean_val(meta.get("title", ""))
        speaker = clean_val(meta.get("all_speakers", ""))
        chunk   = str(meta.get("chunk", "") or "")
        year    = clean_val(meta.get("published_year", ""))
        langs   = clean_val(meta.get("available_lang", ""))  # useful if asked about language

        context_list.append({
            "talk_id": clean_val(meta.get("talk_id", "")),
            "title": title,
            "chunk": chunk,
            "score": round(score, 4),
        })

        context_text += (
            f"talk_id: {clean_val(meta.get('talk_id',''))}\n"
            f"title: {title}\n"
            f"speaker: {speaker}\n"
            f"year: {year}\n"
            f"available_lang: {langs}\n"
            f"transcript_chunk:\n{chunk}\n"
            f"---\n"
        )

    # ---- System Prompt (keep professor's required text; avoid smart quotes) ----
    sys_prompt = (
        "You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). "
        "You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. "
        "If the answer cannot be determined from the provided context, respond: \"I don't know based on the provided TED data.\" "
        "Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.\n\n"
        "Output format:\n"
        "- First: the direct answer only.\n"
        "- Then at the end: EXPLANATION: (evidence from the provided context only).\n"
    )

    # ---- Format rule ----
    if n_exact and n_exact > 1:
        format_rule = (
            f"Return exactly {n_exact} DISTINCT talks (not multiple chunks from the same talk). "
            "List the talk titles (and speakers if requested). Then add EXPLANATION."
        )
    else:
        format_rule = (
            "Provide the requested information in the direct answer. "
            "Then add EXPLANATION at the end."
        )

    user_prompt = (
        "Context:\n--- CONTEXT START ---\n"
        f"{context_text}"
        "--- CONTEXT END ---\n\n"
        f"Question: {question}\n\n"
        f"Constraint: {format_rule}"
    )

    chat = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "response": chat.choices[0].message.content,
        "context": context_list,
        "Augmented_prompt": {"System": sys_prompt, "User": user_prompt},
    }
