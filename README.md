# Genshin Impact Hybrid QA System

**Genshin Impact Hybrid QA System** is a multi-source hybrid question-answering system built to explore the rich narrative of **Genshin Impact**.  
It integrates structured and unstructured data sources to answer lore-related queries with high factual accuracy and contextual nuance.

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

## Target Users & Use Case

This system is designed for Genshin Impact fans who want to **quickly understand game lore** without spending hours reading in-game text or wiki pages.

Use cases include:
- Casual players who skip cutscenes but are curious about the deeper story  
- Lore-focused content creators or YouTubers seeking factual summaries  
- Game developers or researchers studying AI's impact on interactive storytelling

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
```mermaid
graph TD
  Q[User Query] --> RW[Rewrite]
  RW --> IC[Intent Classification]
  IC --> DQ[Decompose & Expand]
  DQ -->|To KG| KG[Knowledge Graph]
  DQ -->|To RAG| RAG[Chroma + Gemini]
  DQ -->|To BM25| BM[BM25 Retriever]
  KG --> SY[Gemini Synthesis]
  RAG --> SY
  BM --> SY
  SY --> A[Answer Output]
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
## Prompt Engineering Examples

This system uses Gemini 1.5 Pro to guide the following prompt-based stages: query rewriting, intent classification, decomposition, and query expansion.

### Rewrite Query

```python
prompt = f"""
Rewrite the following informal or unclear user query into a formal, unambiguous search query.

Original: "{query}"

Rewritten:
"""
```
###  Classify Intent
```python
prompt = f"""
You are a Genshin Impact search assistant.
Classify the user query into one of the following intents:
["character_relationship", "plot_event", "location_info", "quest_requirement", "general_knowledge"]

Query: "{query}"

Return only the intent.
"""
```
###   Decompose Complex Queries
```python
prompt = f"""
You are a smart assistant. Break the user's complex query into 2–4 simpler sub-questions.

Query: "{query}"

Return the result as a JSON list of strings only, without any explanation or markdown.
"""
```
###   Expand Query Variants
```python
prompt = f"""
You are a Genshin Impact lore expert.

Please expand the following user question into **3 to 5 semantically related search queries** to increase retrieval coverage.

Original question: "{question}"

Requirements:
- Include paraphrases, synonyms, or alternate phrasings.
- Keep each query clear and self-contained.
- Return the output as a **pure JSON array of strings**, and nothing else.
- Format strictly like: ["query1", "query2", "query3", ...]
"""
```

---

## Project Structure

```plaintext
genshin-lore-qa-neo4j-rag-bm25/
├── src/
│   ├── data_extraction.py         # Web crawler & wiki text fetcher
│   ├── triple_extraction.py       # Gemini-based relation triplet extraction
│   ├── kg_builder.py              # Neo4j schema + node/edge ingestion
│   ├── rag_builder.py             # Text chunking + vector store (Chroma)
│   ├── bm25_builder.py            # BM25 retriever setup
│   ├── qa_pipeline.py             # Full multi-retriever pipeline
│   └── utils.py                   # Rewrite / classify / expand utilities
├── notebooks/
│   └── genshin_qa_demo.ipynb      # End-to-end demo (query → answer)
├── genshin_chroma/                # Output vector store (ChromaDB)
├── data/                          # Demo output data (KG nodes, edges, etc.)
├── .env.example                   # API & DB credentials template
├── requirements.txt               # Python dependencies
└── README.md                      # You're reading it!
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
```bash
cp .env.example .env
# Fill in your GOOGLE_API_KEY and Neo4j credentials
```

### 3. Launch the notebook
```bash
jupyter notebook notebook/genshin_qa_demo.ipynb
```

---

## Design Rationale

### Why Hybrid Retrieval?

| Method     | Strength                                | Use Case Example                            |
|------------|-----------------------------------------|----------------------------------------------|
| KG (Neo4j) | Precise, structured, interpretable       | "Who does Zhongli know?"                     |
| RAG        | Handles paraphrasing + long-form queries | "Tell me the story behind Rex Lapis"        |
| BM25       | Fast fallback for keyword-heavy queries  | "Liyue act III ending explained"            |

- **Chunking Strategy**: `RecursiveCharacterTextSplitter`  
- **Embedding Model**: Gemini 1.5 Pro via `google-generativeai`  
- **Evaluation Method**: Manual comparison across KG, RAG, and BM25 retrieval outputs

---

## Outcomes & Learnings

- Designed an end-to-end QA system combining LLMs, classical information retrieval, and structured knowledge graphs  
- Practiced prompt engineering for Gemini with a focus on factual grounding  
- Gained hands-on experience with LangChain’s multi-retriever orchestration  
- Built a practical, demo-ready artifact for my AI-focused product management portfolio

---

## Limitations & Future Work

While the current system demonstrates the potential of hybrid retrieval in gaming QA, there are several limitations and areas for future improvement:

### Accuracy & Token Efficiency
- The accuracy of RAG and BM25 components can still be improved.
- Current prompts may result in large token sizes, which affects latency and cost.
- Future work will explore prompt tuning and context selection strategies to balance factual grounding and efficiency.

### Language & Content Scope
- The system currently supports only **English** and focuses on **Archon Quests**.
- To scale to all in-game storylines (e.g., character quests, events) and support **multilingual queries** (e.g., Japanese, Chinese), the pipeline will need:
  - Multilingual embeddings
  - Efficient document indexing and retrieval filtering

### Usability & Deployment
- The current version runs as a local Jupyter notebook.
- Future plans include deploying a user-friendly interface using **Gradio** or **FastAPI** and hosting the system on **Cloud Run or Hugging Face Spaces**, making it accessible to fans and researchers without coding knowledge.

---

## About Me

A product manager with a background in AI product development, NLP, and data strategy.  
This project is part of my portfolio to demonstrate applied generative AI and information retrieval techniques for real-world use cases.

📎 Connect with me on [LinkedIn](https://www.linkedin.com/in/chia-wei-chang-94060b1a0/)  
📂 See more of my work on [GitHub](https://github.com/changch223)


