
# qa_pipeline.py

import re, json
from typing import List
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 需要在模块顶先： genai.configure(api_key=API_KEY); client = genai.Client(api_key=API_KEY)


#  3.3 查詢重寫（Query Rewriting）
# 使用者輸入口語、模糊或錯誤的查詢時，LLM 自動將其重寫為適合檢索的清晰查詢。
def rewrite_query_for_search(query: str) -> str:
    prompt = f"""
Rewrite the following informal or unclear user query into a formal, unambiguous search query.

Original: "{query}"

Rewritten:
"""
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
    return response.text.strip()



# Classifies the intent of a query into one of five categories for more targeted processing
def classify_intent_llm(query: str) -> str:
    prompt = f"""
You are a Genshin Impact search assistant.
Classify the user query into one of the following intents:
["character_relationship", "plot_event", "location_info", "quest_requirement", "general_knowledge"]

Query: "{query}"

Return only the intent.
"""
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
    return response.text.strip()


# Decomposes complex queries into simpler sub-questions for more effective retrieval
# 使用者提出複雜問題時，LLM 拆成多個小問題，有助於提高 RAG 召回率與精準度。
def decompose_query(query: str) -> list[str]:
    prompt = f"""
You are a smart assistant. Break the user's complex query into 2–4 simpler sub-questions.

Query: "{query}"

Return the result as a JSON list of strings only, without any explanation or markdown.
"""
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
    
    # 清理可能含有 ```json 或其他 markdown 語法
    cleaned = re.sub(r"```json|```", "", response.text.strip())
    
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("❌ JSON parsing error:", e)
        print("🔎 Raw response:", response.text)
        return [query]  # fallback: 回傳原始查詢

# Expands a given query into several related phrasings to improve recall in retrieval
def expand_query_with_llm(question: str) -> List[str]:
    import json

    prompt = f"""
You are a Genshin Impact lore expert.

Please expand the following user question into **3 to 5 semantically related search queries** to increase retrieval coverage.

Original question: "{question}"

Requirements:
- Include paraphrases, synonyms, or alternate phrasings.
- Keep each query clear and self-contained.
- Return the output as a **pure JSON array of strings**, and nothing else.
- Format strictly like: ["query1", "query2", "query3", ...]

Now return only the JSON list:
"""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    raw = response.text.strip()
    
    # First try: clean JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Second try: extract JSON-like list manually
    try:
        import re
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass

    # Fallback
    return [question]

# Selects the most semantically diverse queries based on cosine similarity of embeddings
def pick_diverse_queries(queries: List[str], embedding_model, top_k: int = 5) -> List[str]:
    # Generate embeddings for each query
    embeddings = embedding_model.embed_documents(queries)

    # Calculate cosine similarity between all pairs
    sim_matrix = cosine_similarity(embeddings)

    # Mask out self-similarity
    np.fill_diagonal(sim_matrix, -1)

    # Compute average similarity to all other queries
    avg_sim = np.mean(sim_matrix, axis=1)

    # Return the k queries with lowest average similarity (most unique)
    diverse_indices = np.argsort(avg_sim)[:top_k]
    return [queries[i] for i in diverse_indices]


def generate_answer_from_sections(question, kg_section, rag_section, bm25_section) -> str:
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
You are a professional narrative writer and expert in Genshin Impact lore.

Based on the provided structured knowledge graph and retrieved story excerpts, your task is to **retell the complete story** relevant to the topic below. Use the input materials as your only sources.

You must:
- Use **only** the provided materials. Do not invent or guess any details.
- Write in **clear, formal, and engaging prose** suitable for a lore archive or detailed summary.
- Organize the story in **chronological order** (or logical topic groups if time is unclear).
- Incorporate **key characters, locations, items, and events** naturally into the story.
- Provide rich descriptions while remaining strictly faithful to the source.

If some information is unclear or incomplete, acknowledge that appropriately.

---

# Topic:
{question}

---

# Knowledge Graph Summary:
{kg_section}

---

# Retrieved Narrative Excerpts:
{rag_section}

---
# BM25 Retrieved Info:
{bm25_section}

---

# Story (Long with detail):
"""
    response = model.generate_content(prompt)
    return response.text.strip()


