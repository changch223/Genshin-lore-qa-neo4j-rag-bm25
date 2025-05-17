"""
Genshin Impact Triple Extraction with Gemini

Overview:
This script is designed to extract semantic triples from Genshin Impact story texts using Gemini 1.5 Pro.
It processes long story texts by splitting them into manageable chunks and prompting the model to return structured
(subject, relation, object) triples suitable for knowledge graph construction.

The model is guided with strict formatting rules and allowed relation types to ensure consistency and factuality.
The triples can be validated via Pydantic and optionally stored or visualized in downstream knowledge systems.

Modules:
- Triple: A Pydantic BaseModel that defines the triple structure
- generate_triples_batched(): Splits input text, sends it to Gemini, and parses triple results across all chunks
"""

from typing import List
from pydantic import BaseModel
from google import genai

# Define the schema for a semantic triple using Pydantic.
# Each triple represents a factual statement: (subject, relation, object)
class Triple(BaseModel):
    subject: str
    relation: str
    object: str

def generate_triples_batched(text: str, client: genai.Client, model: str="models/gemini-1.5-pro", chunk_size: int=5000) -> List[Triple]:
    """
    Extract semantic triples from a long text using Gemini API in batches.

    Args:
        text (str): The full story text to be processed.
        client (genai.Client): Gemini API client.
        model (str): Model name (default: "models/gemini-1.5-pro").
        chunk_size (int): Maximum number of characters per prompt chunk.

    Returns:
        List[Triple]: A list of extracted semantic triples.
    """
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Splitting into {len(chunks)} chunk(s)...")

    all_triples: List[Triple] = []
    for idx, chunk in enumerate(chunks, 1):
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
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=list[Triple],
                ),
            )
            triples = resp.parsed or []
            all_triples.extend(triples)
        except Exception as e:
            print(f"Error in chunk {idx+1}: {e}")
    
    print(f"✅ Total extracted triples: {len(all_triples)}")
    return all_triples
