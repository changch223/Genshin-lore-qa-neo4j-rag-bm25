"""
BM25 Keyword-Based Retriever Builder

Overview:
This script builds a keyword-based retriever using the BM25 algorithm from a local `.txt` file.
It splits the raw text into overlapping chunks, converts them into LangChain Document objects,
and constructs a `BM25Retriever` to support sparse information retrieval.

Use Cases:
- Quick and lightweight retrieval when semantic embedding is not necessary
- Backup retriever in hybrid RAG systems (e.g., fallback when vector search fails)
- Local keyword-based document search with zero external API calls

Modules:
- build_bm25_retriever_from_txt(): Loads a text file, chunks it, and builds a retriever
"""

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from typing import List

def build_bm25_retriever_from_txt(txt_path: str, k: int = 3) -> BM25Retriever:
    """
    Build a BM25-based keyword retriever from a local plain text file.

    Args:
        txt_path (str): Path to the .txt file containing the source text.
        k (int): Number of top documents to return during retrieval (default is 3).

    Returns:
        BM25Retriever: A keyword-based retriever that ranks documents using BM25 scoring.
    """
    
    # Step 1: Load content
    with open(txt_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(raw_text)

    # Step 3: Convert to LangChain Documents
    documents = [
        Document(page_content=chunk, metadata={"source": f"chunk_{i}"})
        for i, chunk in enumerate(chunks)
    ]

    # Step 4: Create BM25Retriever
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k

    return retriever
