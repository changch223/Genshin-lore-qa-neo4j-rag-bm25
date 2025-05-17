"""
Genshin Impact Knowledge Graph Builder

Overview:
This script automates the process of building a directed knowledge graph from Genshin Impact wiki pages.
It extracts semantic triples using the Gemini API, constructs a directed graph with NetworkX, and exports the results 
as both GraphML (for visualization) and CSV (for downstream processing or Neo4j import).

Features:
- Uses `generate_triples_batched()` to extract (subject, relation, object) triples from wiki text
- Builds a directed graph using NetworkX with labeled edges
- Exports the graph to GraphML format
- Writes extracted nodes and edges to CSV to minimize memory usage during batch crawling

Modules:
- build_graph(): Converts a list of triples into a NetworkX directed graph
- export_graphml(): Saves the graph to a GraphML file
- crawl_and_write_csv(): Fetches wiki text, extracts triples, and writes to CSV files incrementally
"""

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
    """
    Build a directed graph from a list of semantic triples.

    Args:
        triples (List[Triple]): List of extracted (subject, relation, object) triples.

    Returns:
        nx.DiGraph: A directed graph where nodes are entities and edges are labeled by relation types.
    """
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
    Export a directed graph to a GraphML file.

    Args:
        G (nx.DiGraph): The graph to export.
        path (str): Output file path for the GraphML.
    """
    nx.write_graphml(G, path)
    return print(f"Exported {path}")


def crawl_and_write_csv(urls: List[str], nodes_csv: str, edges_csv: str):
    """
    Crawl a list of wiki URLs, extract triples, and write nodes/edges to CSV files incrementally.

    This is memory-efficient and suitable for processing a large number of pages.

    Args:
        urls (List[str]): List of wiki page URLs.
        nodes_csv (str): Path to output CSV file for nodes.
        edges_csv (str): Path to output CSV file for edges.
    """
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