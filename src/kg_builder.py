import csv
import networkx as nx
from typing import List, Tuple, Set
from triple_extraction  import generate_triples_batched, Triple
from data_extraction    import fetch_wiki_text
from google import genai
import os

API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# Build directed graph
def build_graph(triples: List[Triple]) -> nx.DiGraph:
    """從 Triple 列表建圖"""
    # Filter invalid triples
    clean_triples = [
        t for t in triples
        if t.subject.strip() and t.relation.strip() and t.object.strip()
    ]

    # Build directed graph
    G = nx.DiGraph()
    for t in clean_triples:
        G.add_edge(t.subject, t.object, relation=t.relation)

    return G


# Export to GraphML for future use
def export_graphml(G: nx.DiGraph, path: str):
    """
    將圖存成 GraphML，並印出實際使用的檔案路徑
    """
    nx.write_graphml(G, path)
    return print(f"Exported {path}")


def crawl_and_write_csv(urls: List[str], nodes_csv: str, edges_csv: str):
    """逐頁抓取 Wiki + 即時寫入 CSV（避免記憶體爆掉）"""
    written_nodes: Set[str] = set()

    with open(nodes_csv, "w", newline="", encoding="utf-8") as f_nodes, \
         open(edges_csv, "w", newline="", encoding="utf-8") as f_edges:

        node_writer = csv.DictWriter(f_nodes, fieldnames=["id", "label"])
        edge_writer = csv.DictWriter(f_edges, fieldnames=["source", "target", "relation"])
        node_writer.writeheader()
        edge_writer.writeheader()

        for url in urls:
            print("🔍 Processing:", url)

            text = fetch_wiki_text(url)
            if not text.strip():
                print("⚠️ Empty page:", url)
                continue

            triples = generate_triples_batched(text, client)

            for t in triples:
                subj = t.subject.strip()
                obj = t.object.strip()
                rel  = t.relation.strip()

                for entity in (subj, obj):
                    if entity not in written_nodes:
                        node_writer.writerow({"id": entity, "label": "Entity"})
                        written_nodes.add(entity)

                edge_writer.writerow({"source": subj, "target": obj, "relation": rel})

    print(f"✅ Done! Saved to {nodes_csv} & {edges_csv}")