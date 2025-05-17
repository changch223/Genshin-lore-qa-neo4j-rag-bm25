# knowledge_graph.py

from pydantic import BaseModel
from typing import List, Tuple
import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt

# Define Triple schema using Pydantic
class Triple(BaseModel):
    subject: str
    relation: str
    object: str


# Gemini batched semantic triple extraction from story text
def generate_triples_batched(text: str, chunk_size: int = 5000) -> List[Triple]:
    """
    分块调用 Gemini API，返回所有解析后的 Triple 列表。
    """
    # 需要先在模块顶配置好： genai.configure(api_key=API_KEY); client = genai.Client(api_key=API_KEY)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Splitting into {len(chunks)} chunk(s)...")

    all_triples = []

    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}")

        prompt = [
            {
                "text": (
                    "You are a knowledge graph construction expert.\n"
                    "Please extract all semantic triples (subject, relation, object) from the following *Genshin Impact* story text.\n\n"
                    "Each triple must represent a factual statement in the story, such as a character participating in a quest, "
                    "an event occurring in a location, or an item being possessed by a person.\n\n"
                    "Identify and include the following types of nodes:\n"
                    "- Characters (e.g., Traveler, Paimon, Venti)\n"
                    "- Locations (e.g., Mondstadt, Liyue Harbor)\n"
                    "- Items (e.g., Vision, Cleansing Bell)\n"
                    "- Events or Quests (e.g., Prologue Act I, Stormterror Incident)\n"
                )
            },
            {
                "text": (
                    "Only use the following relation types (case-sensitive):\n"
                    "- knows\n"
                    "- member_of\n"
                    "- occurs_in\n"
                    "- has\n"
                    "- participates_in\n"
                    "- opposes\n"
                    "- before\n"
                    "- after\n"
                    "- has_attribute\n"
                    "- from\n"
                    "- plays_role\n"
                    "- includes   # use for connecting an event to its sub-events or scenes"
                )
            },
            {
                "text": (
                    "Output the result as a **pure JSON array** of triples, and do NOT include any explanation or commentary.\n"
                    "Here is an example format:\n\n"
                    "[\n"
                    "  [\"Traveler\", \"from\", \"Teyvat\"],\n"
                    "  [\"Traveler\", \"participates_in\", \"Prologue Act I\"],\n"
                    "  [\"Prologue Act I\", \"before\", \"Prologue Act II\"],\n"
                    "  [\"Stormterror Incident\", \"occurs_in\", \"Mondstadt\"],\n"
                    "  [\"Traveler\", \"participates_in\", \"Stormterror Incident\"],\n"
                    "  [\"Prologue Act I\", \"includes\", \"Stormterror Incident\"]\n"
                    "]"
                )
            },
            {
                "text": "Text:\n" + chunk
            }
        ]

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=list[Triple],
                ),
            )
            triples = response.parsed or []
            all_triples.extend(triples)
        except Exception as e:
            print(f"Error in chunk {idx+1}: {e}")

    print(f"✅ Total extracted triples: {len(all_triples)}")
    return all_triples

def visualize_graph(triples: List[Triple]) -> None:
    """
    用 NetworkX + matplotlib 画出有向知识图谱。
    """
    # Build directed graph
    G = nx.DiGraph()
    for t in clean_triples:
        G.add_edge(t.subject, t.object, relation=t.relation)

    # Export to GraphML for future use
    nx.write_graphml(G, "genshin_story_knowledge.graphml")
    print("Exported genshin_story_knowledge.graphml")

    # Plot the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1200, font_size=10, arrowsize=15)

    # Show relation labels on edges
    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    plt.title("Genshin Impact Knowledge Graph")
    plt.axis("off")
    plt.show()

def export_graphml(G: nx.Graph, path: str) -> None:
    nx.write_graphml(G, path)

def load_graphml(path: str) -> nx.Graph:
    return nx.read_graphml(path)

def query_graph(G: nx.Graph, entity: str) -> List[Tuple[str, str, str]]:
    """
    返回所有与给定实体相连的 (subject, relation, object) 三元组。
    """
    results = []
    for u, v, data in G.edges(data=True):
        if u == entity or v == entity:
            results.append((u, data.get("relation",""), v))
    return results