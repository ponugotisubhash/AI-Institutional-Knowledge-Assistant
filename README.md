# AI Institutional Knowledge Assistant

## Overview

AI Institutional Knowledge Assistant is a Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask natural language questions.

The system combines semantic retrieval, reranking, answer generation, grounding validation, audit logging, and observability monitoring to provide accurate and explainable responses.

---

## Features

- PDF Document Ingestion
- Text Chunking
- Semantic Search
- Keyword Search
- FAISS Vector Retrieval
- CrossEncoder Re-ranking
- Gemini 2.5 Flash Integration
- Grounding Validation
- Hallucination Detection
- Audit Logging
- Observability Dashboard

---

## System Architecture

### Solution Flow

User Uploads PDF
↓
Knowledge Extraction
↓
Semantic Retrieval
↓
Answer Generation
↓
Grounding Validation
↓
Monitoring Dashboard

---

## Technology Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Re-ranking | CrossEncoder |
| LLM | Gemini 2.5 Flash |
| Database | SQLite |

---

## Technical Design Decisions

| Component | Why Chosen |
|------------|------------|
| MiniLM | Lightweight and fast embeddings |
| FAISS | Efficient semantic retrieval |
| CrossEncoder | Improves relevance ranking |
| Gemini 2.5 Flash | Fast response generation |
| SQLite | Lightweight audit logging |
| Streamlit | Rapid AI application development |

---

## Key Metrics

- Grounding Score
- Retrieval Time
- LLM Response Time
- PASS / FAIL Validation
- Audit Records

---

## Future Roadmap

- Multi-PDF Support
- Source Citations
- Authentication
- Cloud Deployment
- Enterprise Knowledge Platform

---

## Author

Subhash P

Master of Science in Data Science

University of North Texas
