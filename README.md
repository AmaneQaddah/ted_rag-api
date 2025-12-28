# TED Talk RAG API

This project implements a Retrieval-Augmented Generation (RAG) system over a TED Talks dataset.
The system answers questions strictly based on retrieved TED transcript chunks and metadata.

## Live Deployment
- Base URL: https://ted-rag-api-eight.vercel.app
- Swagger Docs: https://ted-rag-api-eight.vercel.app/docs

## Main Endpoint

### POST /api/prompt
Send a natural language question to query the system.

**Request Body**
```json
{
  "question": "Your question here"
}
