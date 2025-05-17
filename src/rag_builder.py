"""
Genshin Impact RAG Vectorstore Builder using Gemini Embeddings

Overview:
This script builds a Retrieval-Augmented Generation (RAG) vector store using content extracted from
Genshin Impact's Archon Quest wiki pages. It splits the wiki content into overlapping text chunks,
generates vector embeddings using Gemini API (`text-embedding-004`), and stores them in a local Chroma vector database.

Features:
- Uses LangChain's Document format and Chroma for vector database storage
- Applies recursive character-based text splitting for chunking
- Deduplicates chunks using MD5 hash for storage efficiency
- Implements a custom Embedding class (`GeminiEmbeddings`) for document and query embedding using Gemini

Modules:
- GeminiEmbeddings: Wrapper to call Gemini's embedding API for documents and queries
- build_rag_vectorstore(): Fetches content, chunks it, embeds it, and saves it to Chroma DB
"""

import hashlib
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from typing import List
from google.genai import types
from google import genai
from data_extraction import fetch_act_urls, fetch_page_text_with_headings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

EMBED_MODEL = "models/text-embedding-004"

class GeminiEmbeddings(Embeddings):
    """
    Custom embedding class that uses Google's Gemini API to generate vector representations
    for both documents (for indexing) and queries (for searching).
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates document embeddings in batches for efficient retrieval storage.

        Args:
            texts (List[str]): List of text chunks to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        all_embs = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    title="; ".join([t[:50] for t in batch])
                )
            )
            all_embs.extend(e.values for e in resp.embeddings)
        return all_embs

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a single embedding vector for a search query.

        Args:
            text (str): The user query.

        Returns:
            List[float]: Embedding vector of the query.
        """
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            )
        )
        return resp.embeddings[0].values
    

def build_rag_vectorstore(persist_path="./genshin_chroma") -> Chroma:
    """
    Fetches and embeds Genshin Impact wiki content to create a Chroma vector store
    for use in Retrieval-Augmented Generation (RAG) applications.

    Args:
        persist_path (str): Directory path to save the Chroma database.

    Returns:
        Chroma: The constructed vector store object.
    """
    urls = fetch_act_urls()
    docs: List[Document] = []
    seen_hashes = set()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )

    for url in urls:
        page_docs = fetch_page_text_with_headings(url)
        for doc in page_docs:
            section = doc.metadata.get("section_title", "")
            content = f"### {section}\n{doc.page_content}" if section else doc.page_content
            chunks = splitter.split_text(content)
            for idx, chunk in enumerate(chunks):
                h = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": doc.metadata["source"],
                                "chunk_index": idx,
                                "section_title": section
                            }
                        )
                    )

    print(f"✅ Total unique document chunks: {len(docs)}")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GeminiEmbeddings(),
        persist_directory=persist_path
    )
    vectorstore.persist()
    print(f"✅ Vector store saved to {persist_path}")
    return vectorstore
