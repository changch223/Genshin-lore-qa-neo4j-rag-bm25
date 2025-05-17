# LoreLens: Genshin Impact Hybrid QA System

This project is a multi-source question-answering system that explores **Genshin Impact's lore** using a hybrid of structured and unstructured data.  
Built with **Neo4j**, **LangChain**, **Google Gemini API**, and **BM25**, it retrieves facts from:

- **Knowledge Graph** (Neo4j)
- **RAG** (Retrieval-Augmented Generation) using Gemini Embeddings + ChromaDB
- **BM25** keyword-based retrieval

---

## What It Can Do

- Rewrite informal questions into search-optimized queries
- Classify query intent (e.g., character relationship, plot event, location info)
- Search across KG / semantic / keyword sources
- Generate high-quality summaries using only retrieved, verifiable facts
- visualize entity relationships (e.g., “Who knows Paimon?”)

> **Example Query:**  
> _"What's Paimon's deal with Zhongli?"_  
> → Answer generated from a triple-source pipeline: KG + RAG + BM25

---

## ⚙Tech Stack

| Layer         | Tool / Framework               |
|---------------|--------------------------------|
| LLM           | Gemini 1.5 Pro (Google API)    |
| RAG Retrieval | ChromaDB (via LangChain)       |
| Keyword Search| BM25Retriever                  |
| Graph Search  | Neo4j + LangChain Cypher QA    |
| Framework     | LangChain                      |
| Web Scraping  | BeautifulSoup + requests       |
| Graph Drawing | NetworkX + Matplotlib          |

---

---

## Project Structure

```plaintext
genshin-lore-qa/
├── src/
│   ├── data_extraction.py
│   ├── triple_extraction.py
│   ├── kg_builder.py
│   ├── rag_builder.py
│   ├── bm25_builder.py
│   ├── qa_pipeline.py
│   └── utils.py
├── notebook/
│   └── genshin_qa_demo.ipynb
├── .env.example
└── README.md
