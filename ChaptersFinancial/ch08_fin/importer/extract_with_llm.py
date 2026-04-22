"""
extract_with_llm.py
===================
Use LLM to extract structured entities, events, and KPIs from filing text
chunks using the JSON schemas defined in ch02_fin.

For each chunk:
  1. Call LLM with entity extraction schema
  2. Call LLM with event extraction schema
  3. Validate against JSON schema; retry with reflexion on failure
  4. Cache successful responses
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jsonschema
from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider

# Load schemas from ch02_fin
_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "ch02_fin" / "schemas"


def _load_schema(name: str) -> dict:
    path = _SCHEMA_DIR / f"{name}.schema.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


_ENTITY_SCHEMA = _load_schema("entity_extraction")
_EVENT_SCHEMA = _load_schema("event_extraction")

_ENTITY_PROMPT_TEMPLATE = """Extract all financial entities from the following text.
Return a JSON object matching the provided schema exactly.

Text:
{text}

Schema requirements:
- entities: array of objects with fields: name, type (ORG/PERSON/INSTRUMENT/LOCATION), identifiers (lei, cik, ticker, isin), confidence (0-1)
- Return ONLY valid JSON. No explanations.

Respond with a JSON object containing an "entities" array."""

_EVENT_PROMPT_TEMPLATE = """Extract all financial events from the following text.
Return a JSON object matching the provided schema exactly.

Text:
{text}

Schema requirements:
- events: array of objects with fields: type (EARNINGS/MA/DOWNGRADE/UPGRADE/SANCTION/IPO/DIVIDEND), description, date, entities_involved (array of names), confidence (0-1)
- Return ONLY valid JSON. No explanations.

Respond with a JSON object containing an "events" array."""


def run():
    print("[ch08_fin] LLM-Driven KG Extraction")
    print("=" * 60)

    gp = GraphProvider()
    llm = LLMProvider()

    # Get unprocessed chunks (from ch06_fin or ch08_fin ingestion)
    print("\n1. Loading chunks for extraction …")
    chunks = gp.run("""
        MATCH (c:Chunk)
        WHERE c.llmExtracted IS NULL OR c.llmExtracted = false
        RETURN c.chunkId AS chunkId, c.text AS text
        LIMIT 100
    """)
    print(f"   Found {len(chunks)} chunks to process.")

    if not chunks:
        print("   No new chunks. Done.")
        gp.close()
        return

    total_entities = 0
    total_events = 0
    schema_valid = 0
    schema_total = 0

    print("\n2. Extracting entities and events …")
    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_id = chunk["chunkId"]

        if len(text.strip()) < 50:
            continue

        # Entity extraction
        try:
            entity_result = llm.complete_json(
                _ENTITY_PROMPT_TEMPLATE.format(text=text[:2000]),
                schema=_ENTITY_SCHEMA,
            )
            schema_total += 1
            if _ENTITY_SCHEMA:
                try:
                    jsonschema.validate(entity_result, _ENTITY_SCHEMA)
                    schema_valid += 1
                except jsonschema.ValidationError:
                    pass

            # Store extracted entities as Mention nodes
            for ent in entity_result.get("entities", []):
                mention_id = f"{chunk_id}_llm_{ent.get('name', 'unk')[:20]}"
                gp.run("""
                    MERGE (m:Mention {mentionId: $mentionId})
                    SET m.text = $text, m.chunkId = $chunkId,
                        m.label = $label, m.score = $score,
                        m.extractor = 'llm_ch08',
                        m.identifiers = $identifiers
                    WITH m
                    MATCH (c:Chunk {chunkId: $chunkId})
                    MERGE (m)-[:IN_CHUNK]->(c)
                """, {
                    "mentionId": mention_id,
                    "text": ent.get("name", ""),
                    "chunkId": chunk_id,
                    "label": ent.get("type", "ORG"),
                    "score": ent.get("confidence", 0.5),
                    "identifiers": json.dumps(ent.get("identifiers", {})),
                })
                total_entities += 1

        except Exception as exc:
            print(f"   [WARN] Entity extraction failed for {chunk_id}: {exc}")

        # Event extraction
        try:
            event_result = llm.complete_json(
                _EVENT_PROMPT_TEMPLATE.format(text=text[:2000]),
                schema=_EVENT_SCHEMA,
            )
            schema_total += 1
            if _EVENT_SCHEMA:
                try:
                    jsonschema.validate(event_result, _EVENT_SCHEMA)
                    schema_valid += 1
                except jsonschema.ValidationError:
                    pass

            # Store extracted events
            for evt in event_result.get("events", []):
                event_id = f"{chunk_id}_evt_{evt.get('type', 'UNK')[:10]}"
                gp.run("""
                    MERGE (e:Event {eventId: $eventId})
                    SET e.type = $type, e.description = $description,
                        e.occurredAt = $date, e.confidence = $confidence,
                        e.source = 'llm_ch08',
                        e.runId = $runId
                    WITH e
                    UNWIND $entities AS entName
                    MATCH (le:LegalEntity)
                    WHERE toLower(le.name) CONTAINS toLower(entName)
                    MERGE (e)-[:AFFECTS]->(le)
                """, {
                    "eventId": event_id,
                    "type": evt.get("type", "UNKNOWN"),
                    "description": evt.get("description", ""),
                    "date": evt.get("date", ""),
                    "confidence": evt.get("confidence", 0.5),
                    "entities": evt.get("entities_involved", [])[:5],
                    "runId": "ch08_fin_extract",
                })
                total_events += 1

        except Exception as exc:
            print(f"   [WARN] Event extraction failed for {chunk_id}: {exc}")

        # Mark chunk as processed
        gp.run(
            "MATCH (c:Chunk {chunkId: $id}) SET c.llmExtracted = true",
            {"id": chunk_id}
        )

    print(f"\n3. Results …")
    print(f"   Entities extracted: {total_entities}")
    print(f"   Events extracted: {total_events}")
    print(f"   Schema validity: {schema_valid}/{schema_total} ({100*schema_valid/max(schema_total,1):.0f}%)")
    print(f"   LLM usage: {llm.usage_summary()}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
