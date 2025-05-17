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
    """使用 Gemini API 的向量嵌入實作"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            )
        )
        return resp.embeddings[0].values
    

def build_rag_vectorstore(persist_path="./genshin_chroma") -> Chroma:
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
