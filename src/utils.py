"""
Genshin Impact Query Optimization Toolkit

Overview:
This module contains a set of utility functions designed to improve question understanding,
query decomposition, and retrieval performance for a Genshin Impact hybrid QA system.

It leverages Gemini 1.5 Pro to rewrite vague queries, classify intent, break down complex questions,
expand phrasing for better recall, and select the most diverse reformulations using semantic similarity.

Functions:
- rewrite_query_for_search(): Rewrites informal or vague queries into formal search-ready form
- classify_intent_llm(): Classifies query intent into one of five categories
- decompose_query(): Splits complex questions into simpler sub-questions
- expand_query_with_llm(): Expands a single query into 3–5 paraphrased search versions
- pick_diverse_queries(): Selects the top-k semantically diverse queries for balanced recall
"""

import re, json
import google.generativeai as genai
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Rewrites vague user queries into clear, formal search queries suitable for retrieval
def rewrite_query_for_search(query: str) -> str:
    """
    Rewrites a user's original query into a more structured and search-friendly version.

    Args:
        query (str): The original, possibly vague user query.

    Returns:
        str: A clear, reformulated search query.
    """
    prompt = f"""
    Rewrite the following informal or unclear user query into a formal, unambiguous search query.

    Original: "{query}"

    Rewritten:
    """
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
    return response.text.strip()

# Classifies the intent of a query into one of five categories for more targeted processing
def classify_intent_llm(query: str) -> str:
    """
    Classifies a query's intent into categories to guide retrieval and answer style.

    Categories:
    - character_relationship
    - plot_event
    - location_info
    - quest_requirement
    - general_knowledge

    Args:
        query (str): The rewritten or raw query.

    Returns:
        str: The predicted intent category.
    """
    prompt = f"""
    You are a Genshin Impact search assistant.
    Classify the user query into one of the following intents:
    ["character_relationship", "plot_event", "location_info", "quest_requirement", "general_knowledge"]

    Query: "{query}"

    Return only the intent.
    """
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
    return response.text.strip()

# Breaks complex or multi-part queries into simpler, focused sub-questions
def decompose_query(query: str) -> List[str]:
    """
    Decomposes a complex user query into 2–4 simpler sub-questions to improve retrieval precision.

    Args:
        query (str): The original query.

    Returns:
        List[str]: A list of simplified sub-questions.
    """
    prompt = f"""
    You are a smart assistant. Break the user's complex query into 2–4 simpler sub-questions.

    Query: "{query}"

    Return the result as a JSON list of strings only, without any explanation or markdown.
    """
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
    
    # Remove any markdown syntax like ```json
    cleaned = re.sub(r"```json|```", "", response.text.strip())
    
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("❌ JSON parsing error:", e)
        print("🔎 Raw response:", response.text)
        return [query]  # fallback to original if parsing fails

# Expands a given query into multiple alternative phrasings to improve retrieval coverage
def expand_query_with_llm(question: str) -> List[str]:
    """
    Expands a question into 3–5 related queries (paraphrases, synonyms, etc.) for broader recall.

    Args:
        question (str): Original user query or sub-question.

    Returns:
        List[str]: A list of semantically similar queries.
    """
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

    # Final fallback: return the original input
    return [question]

# Selects the most diverse queries from a list using cosine similarity on embeddings
def pick_diverse_queries(queries: List[str], embedding_model, top_k: int=5) -> List[str]:
    """
    Picks the most semantically diverse queries using cosine similarity scoring.

    Args:
        queries (List[str]): A list of candidate queries.
        embedding_model: A model with an `embed_documents()` method.
        top_k (int): Number of diverse queries to return.

    Returns:
        List[str]: A list of the top-k most distinct queries.
    """
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