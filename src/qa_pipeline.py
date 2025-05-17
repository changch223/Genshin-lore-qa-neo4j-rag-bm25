"""
Multi-Source QA Pipeline for Genshin Impact Lore

Overview:
This script implements a hybrid multi-source question answering (QA) system over Genshin Impact lore.
It integrates a Neo4j-based knowledge graph (KG), a Gemini-powered LLM, a Chroma vectorstore for RAG,
and a BM25 keyword retriever to answer user questions with structured and unstructured sources.

Main Components:
- GeminiLLM: A custom LangChain-compatible wrapper for Gemini 1.5 Pro
- GraphCypherQAChain: LangChain chain for Cypher-based KG reasoning
- generate_answer_from_sections(): Synthesizes final answers from KG, RAG, and BM25 sources
- run_multi_source_qa(): Main pipeline combining rewriting, classification, retrieval, and generation

Use Cases:
- Explain character relationships
- Trace storylines and quests
- Explore event timelines and hidden links between entities
"""

from typing import Optional, List
from langchain_core.prompts import PromptTemplate
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain.schema.messages import AIMessage
from utils              import rewrite_query_for_search, decompose_query, expand_query_with_llm, pick_diverse_queries, classify_intent_llm
import google.generativeai as genai
import warnings
warnings.filterwarnings("ignore")


def init_gemini_llm(api_key: str):
    genai.configure(api_key=api_key)
    return GeminiLLM

def build_kg_qa_chain(uri, user, pwd, gemini_llm, top_k=5):
    """
    Builds a LangChain GraphCypherQAChain for querying Neo4j with natural language.

    Args:
        uri (str): Neo4j URI
        user (str): Username
        pwd (str): Password
        gemini_llm: LangChain-compatible Gemini LLM
        top_k (int): Number of top Cypher matches to return

    Returns:
        GraphCypherQAChain: Configured QA chain using Gemini and Neo4j
    """
    graph = Neo4jGraph(url=uri, username=user, password=pwd)
    cypher_prompt = PromptTemplate(
    input_variables=["query", "id"],
    template="""
    # Instruction:
    You are an expert at translating natural language questions into Cypher queries. 
    Based on the user's question and the entity ID "{id}", generate the **most semantically appropriate** Cypher query. 

    The graph contains nodes of type `Entity`, connected via `:relation` edges.
    Each `:relation` has a `relation` property indicating the actual relationship (e.g. "knows", "interacts_with", "member_of", "originates_from", etc).

    ## General Guidelines:

    1. **If the query explicitly mentions two entities** (e.g., "Paimon and Zhongli"), search for all paths (not just shortest) between the two using up to 4 hops.
    2. **If the query involves a specific relation type** (e.g., "origin", "creator", "is connected to the Archons"), filter by `r.relation`.
    3. **If the query is about hidden or indirect links** (e.g., "Is Paimon related to Rex Lapis?"), include intermediate nodes and expand depth to 4–5 levels.
    4. Use case-insensitive matching (`id =~ "(?i).*<name>.*"`) when the second entity is vague or undefined.
    5. Prefer this return format: 
    `r.relation AS relation_type, b.id AS node`
    6. Ensure that any `WHERE r.relation =~ ...` clauses appear *after* `UNWIND`, not before.

    ## Examples:

    Q: "What relations does Traveler have?"  
    MATCH (a:Entity {{id:"Traveler"}})-[r:relation]-(b:Entity)  
    RETURN r.relation AS relation_type, b.id AS node

    Q: "Is Paimon related to Morax?"  
    MATCH p = (a:Entity {{id:"Paimon"}})-[*..4]-(b:Entity)  
    WHERE b.id =~ "(?i).*morax.*" OR b.id =~ "(?i).*rex lapis.*" OR b.id =~ "(?i).*zhongli.*"  
    WITH relationships(p) AS rels  
    UNWIND rels AS r  
    WITH r, endNode(r) AS b  
    RETURN r.relation AS relation_type, b.id AS node

    Q: "Paimon's origin and Geo Archon connection?"  
    MATCH p = (a:Entity {{id:"Paimon"}})-[*..4]-(b:Entity)  
    WHERE b.id =~ "(?i).*geo.*" OR b.id =~ "(?i).*archon.*" OR b.id =~ "(?i).*zhongli.*"  
    WITH relationships(p) AS rels  
    UNWIND rels AS r  
    WITH r, endNode(r) AS b  
    RETURN r.relation AS relation_type, b.id AS node

    ---

    # Now generate the Cypher query:

    Question: {query}  
    Cypher:
    """
    )
    qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    You are an expert at reading Cypher query results from a knowledge graph.

    Question: {question}
    Cypher Results:
    {context}

    List **all** nodes connected to the entity, grouped by `relation_type`.
    Each group should list all connected nodes exactly as they appear, without summarizing.
    Just output the structured result clearly.
    """
    )
    return GraphCypherQAChain.from_llm(
        llm=gemini_llm, graph=graph,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        top_k=top_k
    )


# === Custom Gemini LLM Wrapper for LangChain ===
class GeminiLLM(BaseLLM):
    """Custom Gemini model wrapper compatible with LangChain chains."""

    model: str = "models/gemini-1.5-pro"
    temperature: float = 0.2
    max_output_tokens: int = 512

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """Generate responses from Gemini for a list of prompts."""
        responses = []
        gemini = genai.GenerativeModel(self.model)

        for prompt in prompts:
            response = gemini.generate_content(
                [{"role": "user", "parts": [{"text": prompt}]}],
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens
                )
            )
            text = response.text.strip()
            responses.append(ChatGeneration(message=AIMessage(content=text)))

        return LLMResult(generations=[[res] for res in responses])

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"


# === Main function to build and run QA Chain ===
def run_kg_qa(query: str, entity_id: str, api_key: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    genai.configure(api_key=api_key)

    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password
    )

    cypher_prompt = PromptTemplate(
        input_variables=["query", "id"],
        template="""
        # Instruction:
        You are an expert at translating natural language questions into Cypher queries. 
        Based on the user's question and the entity ID "{id}", generate the **most semantically appropriate** Cypher query. 

        The graph contains nodes of type `Entity`, connected via `:relation` edges.
        Each `:relation` has a `relation` property indicating the actual relationship (e.g. "knows", "interacts_with", "member_of", "originates_from", etc).

        ## General Guidelines:

        1. **If the query explicitly mentions two entities** (e.g., "Paimon and Zhongli"), search for all paths (not just shortest) between the two using up to 4 hops.
        2. **If the query involves a specific relation type** (e.g., "origin", "creator", "is connected to the Archons"), filter by `r.relation`.
        3. **If the query is about hidden or indirect links** (e.g., "Is Paimon related to Rex Lapis?"), include intermediate nodes and expand depth to 4–5 levels.
        4. Use case-insensitive matching (`id =~ "(?i).*<name>.*"`) when the second entity is vague or undefined.
        5. Prefer this return format: 
        `r.relation AS relation_type, b.id AS node`
        6. Ensure that any `WHERE r.relation =~ ...` clauses appear *after* `UNWIND`, not before.

        ## Examples:

        Q: "What relations does Traveler have?"  
        MATCH (a:Entity {{id:"Traveler"}})-[r:relation]-(b:Entity)  
        RETURN r.relation AS relation_type, b.id AS node

        Q: "Is Paimon related to Morax?"  
        MATCH p = (a:Entity {{id:"Paimon"}})-[*..4]-(b:Entity)  
        WHERE b.id =~ "(?i).*morax.*" OR b.id =~ "(?i).*rex lapis.*" OR b.id =~ "(?i).*zhongli.*"  
        WITH relationships(p) AS rels  
        UNWIND rels AS r  
        WITH r, endNode(r) AS b  
        RETURN r.relation AS relation_type, b.id AS node

        Q: "Paimon's origin and Geo Archon connection?"  
        MATCH p = (a:Entity {{id:"Paimon"}})-[*..4]-(b:Entity)  
        WHERE b.id =~ "(?i).*geo.*" OR b.id =~ "(?i).*archon.*" OR b.id =~ "(?i).*zhongli.*"  
        WITH relationships(p) AS rels  
        UNWIND rels AS r  
        WITH r, endNode(r) AS b  
        RETURN r.relation AS relation_type, b.id AS node

        ---

        # Now generate the Cypher query:

        Question: {query}  
        Cypher:
        """
    )

    qa_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are an expert at reading Cypher query results from a knowledge graph.

        Question: {question}
        Cypher Results:
        {context}

        List **all** nodes connected to the entity, grouped by `relation_type`.
        Each group should list all connected nodes exactly as they appear, without summarizing.
        Just output the structured result clearly.
        """
    )

    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GeminiLLM(
        callback_manager=CallbackManager(callbacks),
        verbose=True
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        verbose=True,
        top_k=5
    )

    output = chain.invoke({
        "query": query,
        "id": entity_id
    })

    return output


def generate_answer_from_sections(question, kg_section, rag_section, bm25_section, intent) -> str:
    """
    Combines structured KG data, RAG excerpts, and BM25 info to generate a complete narrative answer.

    Args:
        question (str): Original user query
        kg_section (str): Structured graph result
        rag_section (str): Unstructured text from RAG
        bm25_section (str): Fallback keyword match result
        intent (str): Classified intent type

    Returns:
        str: Final narrative-style answer
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    intent_style_map = {
    "character_relationship": "Focus on describing relationships between characters and their development.",
    "plot_event": "Retell the sequence of events in a clear chronological narrative.",
    "location_info": "Emphasize geographic context and environment descriptions.",
    "quest_requirement": "Summarize goals, challenges, and objectives relevant to the quest.",
    "general_knowledge": "Give a well-organized factual overview without unnecessary detail."
    }
    style_hint = intent_style_map.get(intent, "Write in a neutral style suitable for a lore summary.")

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
    # Writing Style Hint (based on intent):
    {style_hint}
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


def run_multi_source_qa(
    query: str,
    entity_id: str,
    chain,               # LangChain KG QA chain
    vectorstore,         # RAG vectorstore
    bm25_retriever,      # BM25 retriever
    embedding_model      # embedding model
):
    """
    End-to-end QA pipeline using KG, RAG, and BM25 sources to produce a reliable answer.

    Steps:
    1. Rewrite query for clarity
    2. Classify the query intent
    3. Decompose complex queries into sub-questions
    4. Expand sub-queries using LLM
    5. Run KG retrieval with GraphCypherQAChain
    6. Perform RAG search in vectorstore
    7. Fallback BM25 keyword retrieval
    8. Fuse results into a coherent story using Gemini LLM

    Args:
        query (str): User question
        entity_id (str): Entity ID (e.g., "Paimon")
        chain: LangChain KG QA chain
        vectorstore: Chroma vector database
        bm25_retriever: Keyword-based fallback retriever
        embedding_model: Used for picking diverse queries

    Returns:
        str: Final answer combining multiple knowledge sources
    """
    print("📥 Original Query:", query)

    # 1. Rewrite
    rewritten = rewrite_query_for_search(query)
    print("📝 Rewritten Query:\n", rewritten)

    # 2. Classify
    intent = classify_intent_llm(rewritten)
    print("🔍 Classified Intent:\n", intent)

    # 3. Decompose
    sub_queries = decompose_query(rewritten)
    print("🧩 Sub-Queries:")
    for q in sub_queries:
        print("  -", q)

    # 4. Expand
    expanded_queries = []
    for sub_q in sub_queries:
        expanded_queries.extend(expand_query_with_llm(sub_q))

    top3_queries = pick_diverse_queries(expanded_queries, embedding_model, top_k=3)

    # 5. KG Retrieval
    kg_results = []
    for q in top3_queries:
        try:
            output = chain.invoke({"query": q, "id": entity_id})
            if isinstance(output, dict) and "text" in output:
                kg_results.append(output["text"])
            elif isinstance(output, str):
                kg_results.append(output)
        except Exception as e:
            print(f"⚠️ KG Query failed for: '{q}'")
            print("🔍 Error:", e)
    kg_result = "\n".join(set(kg_results))

    # 6. RAG Retrieval
    rag_docs = []
    for q in expanded_queries:
        rag_docs.extend(vectorstore.similarity_search(q, k=3))
    rag_result = "\n".join(set(doc.page_content for doc in rag_docs))

    # 7. BM25 Retrieval
    bm25_docs = []
    for q in expanded_queries:
        bm25_docs.extend(bm25_retriever.get_relevant_documents(q))
    bm25_result = "\n".join(set(doc.page_content for doc in bm25_docs))

    # 8. Answer Generation
    final_answer = generate_answer_from_sections(
        question=query,
        kg_section=kg_result,
        rag_section=rag_result,
        bm25_section=bm25_result,
        intent=intent 
    )

    return final_answer
