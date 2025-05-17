# LoreLens: Genshin Impact Hybrid QA System

**Product Manager Portfolio Project – with Generative AI & Retrieval Technologies**

**LoreLens** is a multi-source hybrid question-answering system built to explore the rich narrative of **Genshin Impact**.  
It integrates structured and unstructured data sources to answer lore-related queries with high factual accuracy and contextual nuance.

---

## Why This Project?

In lore-heavy games like Genshin Impact, finding concise, accurate story answers across thousands of wiki pages can be overwhelming.  
As a product manager passionate about AI, I built this system to explore:

- How hybrid retrieval architectures improve user trust and search satisfaction
- How GenAI (Google Gemini) can be grounded in facts from knowledge graphs and vector search
- How to design a GenAI-powered feature end-to-end, from crawling to UX

---

## What It Can Do

- **Rewrite** vague or informal questions into precise, search-optimized queries
- **Classify** query intent (e.g., character relationships, plot events, quest conditions)
- **Retrieve** answers from:
  - Neo4j Knowledge Graph
  - RAG using Gemini Embeddings + ChromaDB
  - BM25 keyword-based search
- **Generate** high-quality summaries grounded in retrieved facts
- **Visualize** entity relationships (e.g., "Who knows Paimon?")

> **Example Query:**  
> _"What's Paimon's deal with Zhongli?"_  
> → Answer is synthesized from KG facts + narrative context + keyword clues

---

## System Pipeline Overview

```text
User Query
   ↓
Rewrite → Intent Classification
   ↓
[KG Search | RAG Search | BM25 Search]
   ↓
Gemini Answer Generation (grounded in retrieved facts)
```

---

## ⚙Tech Stack

| Layer             | Tool / Framework                       |
|-------------------|----------------------------------------|
| LLM               | Gemini 1.5 Pro (Google Generative AI)  |
| RAG Retrieval     | ChromaDB + LangChain                   |
| Keyword Search    | BM25Retriever                          |
| Knowledge Graph   | Neo4j + LangChain GraphCypherQA        |
| Prompt Handling   | LangChain PromptTemplate               |
| Web Scraping      | BeautifulSoup + Requests               |
| Graph Drawing     | NetworkX + Matplotlib                  |
| Embedding Model   | Gemini Embeddings                      |

---

---

## Project Structure

```plaintext
genshin-lore-qa/
├── src/
│   ├── data_extraction.py         # Web crawler & wiki text fetcher
│   ├── triple_extraction.py       # Gemini-based relation triplet extraction
│   ├── kg_builder.py              # Neo4j schema + node/edge ingestion
│   ├── rag_builder.py             # Text chunking + vector store (Chroma)
│   ├── bm25_builder.py            # BM25 retriever setup
│   ├── qa_pipeline.py             # Full multi-retriever pipeline
│   └── utils.py                   # Rewrite / classify / expand utilities
├── notebook/
│   └── genshin_qa_demo.ipynb      # End-to-end demo (query → answer)
├── .env.example                   # API & DB credentials (template)
├── requirements.txt               # Python dependencies
└── README.md                      # You're reading it!
```


