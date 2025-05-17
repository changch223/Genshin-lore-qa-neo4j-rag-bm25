# rag_retriever.py

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import google.generativeai as genai

EMBED_MODEL = "models/text-embedding-004"
# 模块顶需配置 client = genai.Client(api_key=API_KEY)

class GeminiEmbeddings(Embeddings):
    """
    Custom embedding class using Google's Gemini embedding API for documents and queries.
    """
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
        """
        Embeds a user query using Gemini's RETRIEVAL_QUERY mode.
        """
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            )
        )
        return resp.embeddings[0].values

def build_vector_store(docs: List[Document], persist_directory: str) -> Chroma:
    vs = Chroma.from_documents(documents=docs,
                               embedding=GeminiEmbeddings(),
                               persist_directory=persist_directory)
    vs.persist()
    return vs

def rag_similarity_search(vectorstore: Chroma, query: str, k: int) -> List[Document]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)