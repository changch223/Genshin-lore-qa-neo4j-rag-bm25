# LoreLens: Genshin Impact Hybrid QA System

This project is a multi-source question-answering system that explores Genshin Impact's narrative universe using a hybrid of structured and unstructured data. Built with Neo4j, LangChain, Gemini API, and BM25, it retrieves facts from:

- Knowledge Graph (Neo4j)
- RAG with Gemini Embeddings + ChromaDB
- BM25 full-text retrieval

## What It Can Do

- Convert fuzzy user questions into optimized search queries
- Extract and visualize entity relationships from wiki text
- Classify query intent (e.g., character relationship, quest requirement)
- Retrieve facts via graph, semantic, and keyword channels
- Generate formal lore summaries grounded strictly in source data

> Example Query:  
> _"What's Paimon's deal with Zhongli?"_  
> → A structured, source-grounded summary that draws from 3 retrieval methods.

## 🔧 Tech Stack

| Layer        | Tool / Framework        |
|--------------|-------------------------|
| LLM          | Gemini 1.5 Pro (via Google API) |
| Retrieval    | Chroma (RAG), BM25, Neo4j |
| Framework    | LangChain               |
| Graph        | NetworkX + Neo4j        |
| Preprocessing| BeautifulSoup + requests|

## 🗂 Project Structure

