# bm25_retriever.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from typing import List

def load_archon_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_text_into_chunks(text: str,
                           chunk_size: int = 500,
                           chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def build_bm25_retriever(chunks: List[str], k: int = 3) -> BM25Retriever:
    docs = [Document(page_content=c, metadata={"source": f"chunk_{i}"})
            for i, c in enumerate(chunks)]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

def bm25_search(retriever: BM25Retriever, query: str) -> List[Document]:
    return retriever.get_relevant_documents(query)
