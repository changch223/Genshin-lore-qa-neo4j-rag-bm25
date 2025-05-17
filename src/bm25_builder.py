from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from typing import List



def build_bm25_retriever_from_txt(txt_path: str, k: int = 3) -> BM25Retriever:
    """
    從本地 txt 檔建立 BM25 關鍵字檢索器
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
