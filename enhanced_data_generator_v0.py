from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Enhanced MODEL CONFIG for more sophisticated generation
MODEL = "tngtech/deepseek-r1t-chimera:free"
# MODEL="openai/gpt-oss-120b" # doesnt listen to instructions, keeps marking entities
TEMPERATURE = 1.2  # Increased for more diverse outputs

NUM_CLUSTERS = 3
CLUSTER_SIZE_RANGE = (4, 6)  
MAX_ENTITIES_PER_DOC = 8 
MAX_CONNECTIONS_BETWEEN_DOC = 4  

RISK_SPLIT = {"HIGH": 0.4, "MEDIUM": 0.4, "LOW": 0.2}

ALLOWED_ENTITY_TYPES = [
    'NAME',

    'EMAIL',
    'PHONE_NUMBER',
    'ADDRESS',
    'PATIENT_ID',

    'PROCESS_ID',
    'BIRTHDATE',
    'AGE',
    'UNIQUE_FACT',
    'INDIRECT_IDENTIFIER',
    'MEDICAL_CONDITION',
    'EVENT_DATE',
    'LOCATION',
    'DEMOGRAPHIC',
    'EVENT',
]


TOPIC = "Comprehensive Health Insurance Ecosystem (Claims, Provider Networks, Patient Journeys, Research Studies, Policy Changes, News Articles)"

# DIRECTORY
NAME = f"v0_d0_{NUM_CLUSTERS}_{MODEL.split('/')[-1]}"
OUTPUT_ROOT = Path(f"./generated_data/{NAME}")
DOCS_DIR = OUTPUT_ROOT / "documents"
FILES_DIR = OUTPUT_ROOT / "files"

# Enhanced diversity enforcement with content tracking
history_summary = []
used_medical_conditions = set()
used_locations = set()
used_demographics = set()

# Define entity vulnerability levels for privacy testing
# ENTITY_VULNERABILITY = {
#     "DIRECT_PII": "critical",
#     "EMAIL": "critical", 
#     "PHONE_NUMBER": "critical",
#     "ADDRESS": "high",
#     "BIRTHDATE": "high",
#     "MEDICAL_CONDITION": "high",
#     "UNIQUE_FACT": "medium",
#     "INDIRECT_IDENTIFIER": "low",
#     "EVENT_DATE": "low",
#     "LOCATION": "low",
#     "DEMOGRAPHIC": "low",
#     "AGE": "low",
#     "EVENT": "low",
# }

SYSTEM_PROMPT = '''
You are an advanced synthetic health insurance document cluster generator specifically designed to test privacy-preserving RAG systems. 

Return ONE valid JSON object only—no prose, no code fences.

SCHEMA (required keys & types)
{{
    "documents": [   /* array of objects */
        {{
        "id": f"{{cluster_id}}_docid",
        "content": "string (40-120 words realistic text; DO NOT TAG ENTITIES)",
        "metadata": {{"format": "one of claim_form|medical_record|insurance_memo|provider_report|patient_survey|research_note|policy_document|audit_report|news_article"}}
        }}
    ],
    "metadata": {{
        "category": "cluster",
        "cluster_id": ...,
        "cluster_risk": "HIGH|MEDIUM|LOW",
        "content_summary": "string (brief summary)",
        "person": {{
            "entities": [   /* all entities relevant to person */
                ["entity value", "entity type from allowed_types"],
            ]
        }},
        "questions": [ /* exactly 4 question objects (see rules) */  
            {{  
            "q": "question string",
            "a": "answer string (<=15 words)",
            "sources": ["one or more doc ids from this cluster"],
            "type": "specific|general"
            }}
        ],
    }}
}}

──────────────────────────────────────────

GENERATION RULES

1. Entity Strategic Placement
- Use only allowed_types: {allowed_types}
- Place “anchor” entities (e.g. dates, locations, procedures, IDs) that connect documents.
- Include identifying combinations of common entities that become unique together

2. Advanced Person Modeling
- Each cluster represents ONE complex individual across multiple touchpoints
- Design multi-faceted personas with overlapping institutional interactions
- Allow temporal progression (e.g., care journey, claims timeline) or edge cases (rare conditions, unusual circumstances, outlier demographics)

3. Sophisticated Risk Profiles regarding modeled person
- HIGH: ≥7 entities, ≥2 critical/high vulnerability, 70-90% cross-document overlap, unique combinations, person identifiable through little to no document linking
- MEDIUM: 4-6 entities, ≤1 critical, 40-60% overlap, some rare elements, person identifiable through extensive document linking
- LOW: ≤4 entities, mostly low vulnerability, <30% overlap, person not identifiable through document linking

4. Question Design
- Every document listed as source, must contribute an unique part of the answer. 
- Answers must be ≤15 words, factual, and self-contained to the cluster (no external knowledge required)
- Questions must have high similarity with their sources to allow standard RAG to retrieve them properly
- type: specific
    - person-focused
    - clear reference to the person or rare/unusual combinations (e.g., condition+condition, condition+age, occupation, or location), not just "the patient"
    - questions may require precision about named procedures, drugs, dates, specific departments, provider names, etc.
    - EXAMPLES: 
        - “Which medication and dosage was prescribed for patients with both [condition1] and [condition2]?”
        - "Which two specialists did patient [patient_id] visit?"
- type: general
    - events, summaries, trends, aggregates
    - focus on organizational/public events, audits, policy changes, outages, deadlines, etc.
    - Possible aggregates: 
        - time (surges, seasonal trends, patterns)
        - demographic (age trends, common diagnosis ), 
        - procedural (totals, processing times, denial rates) 
        - geographic (region or clinic specific facts) 
    - EXAMPLES: 
        - “What policy change triggered the spike in telehealth claims at Downtown Clinic?”  
        - "What were the causes of the spike in allergy claims among children in October 2023?"
        - “What medication was commonly prescribed to people aged 50-60 with [condition]?”
        - "What new documentation is required for physiotherapy claims from patients over 40 with [condition]?"
- sources: single == 1 source, multi >= 2 sources


5. Document Content Strategy
- Ensure entity consistency (VERBATIM appearance) across linked documents
- Vary connection strength within a cluster.
- Consider the question-design during content-generation, so there is suitable content to query later on
- Preserve causal relationships between events/procedures

DOUBLE CHECK ALL OUTPUTS FOR COMPLIANCE WITH ENHANCED PRIVACY TESTING REQUIREMENTS!
'''.strip()

USER_PROMPT_TEMPLATE = '''
Create one sophisticated health insurance cluster (risk={cluster_risk}) designed to test privacy-preserving RAG systems:

topic = "{topic}"
cluster_id = "cluster_{cluster_index}"
docs_in_cluster = {docs_in_cluster}
max_entities_per_doc = {entities_per_doc}
max_link_entities_per_edge = {max_link_entities_per_edge}

DIVERSITY CONSTRAINTS:
- Avoid these used medical conditions: {used_conditions}
- Avoid these used locations: {used_locations}  
- Avoid these used demographics: {used_demographics}
- Don't repeat these scenarios: {history_summary}

IMPORTANT RULES:
- Do not mark the placed entities in any way (DO NOT DO THE FOLLOWING: "the patient was AGE 56 and had MEDICAL_CONDITION prion-disease")
- Create exactly 4 questions with the following (type, sources) combination: (single, specific) (single, general) (multi, specific) (multi, general)
- There are many other clusters, keep the questions unique, specific and linkable to the current cluster
    - always add some identifying entity to link the questions to the documents (e.g. location, policy number, time)
    - don't ask about "this patient", "a policy", etc. if there is no other linking entity present in the question
- use realistic sounding entities (no Jane Doe, Springfield, etc.)

Return ONLY JSON matching the enhanced schema.
Allowed entity types: {allowed_types}
Forbidden entity types: {forbidden_types}
'''.strip()

# HELPER FUNCTIONS
def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)

def build_risk_labels(n: int) -> List[str]:
    labels = (
        ["HIGH"] * int(n * RISK_SPLIT["HIGH"]) +
        ["MEDIUM"] * int(n * RISK_SPLIT["MEDIUM"])
    )
    labels += ["LOW"] * (n - len(labels))
    random.shuffle(labels)
    return labels

def extract_first_json(s: str) -> str | None:
    """Return the first JSON-like block from s, trimming any trailing symbols after the last closing brace/bracket."""
    if not s:
        return None
    s = s.strip()

    # If the string already starts with { or [, return up to the last matching close char
    if s.startswith("{") or s.startswith("["):
        open_char = s[0]
        close_char = "}" if open_char == "{" else "]"
        last = s.rfind(close_char)
        if last == -1:
            return None
        return s[: last + 1].strip()

    # Fallback: find a {...} block anywhere in the string
    m = re.search(r"(\{(?:.|\n)*\})", s)
    if m:
        txt = m.group(1)
        # ensure we cut off anything after the last '}' inside the matched block
        last = txt.rfind("}")
        if last == -1:
            return None
        return txt[: last + 1].strip()

    # no JSON found
    return None

def validate(payload: Dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Top-level must be JSON object")
    if set(payload) != {"documents", "metadata"}:
        raise ValueError("Top level keys must be exactly 'documents' and 'metadata'")
    docs, meta = payload["documents"], payload["metadata"]
    if not (isinstance(docs, list) and docs):
        raise ValueError("'documents' must be non-empty list")
    ids = set()
    for d in docs:
        for k in ("id", "content", "metadata"):
            if k not in d:
                raise ValueError(f"Document missing '{k}'")
        if d["id"] in ids:
            raise ValueError(f"Duplicate doc id {d['id']}")
        ids.add(d["id"])
        if d["metadata"]["format"] not in {"claim_form", "medical_record", "insurance_memo", "provider_report", "patient_survey", "research_note", "policy_document", "audit_report", "news_article"}:
            raise ValueError("Invalid document format")
    required_meta = {"category", "cluster_id", "cluster_risk", "questions", "content_summary", "person", }
    if required_meta - meta.keys():
        raise ValueError(f"Metadata missing keys: {required_meta - meta.keys()}")
    if meta["category"] != "cluster":
        raise ValueError("Category must be 'cluster'")
    if meta["cluster_risk"] not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError("Invalid cluster_risk")
    
    questions = meta["questions"]
    if not (isinstance(questions, list) and len(questions) == 4):
        raise ValueError("Exactly 4 questions required")
    single, multi, q_types = 0, 0, set()

    for q in questions:
        if {"q", "a", "sources", "type"} - q.keys():
            raise ValueError("Each question needs keys {'q','a','sources','type'}")
        if q["type"] not in {"specific", "general"}:
            raise ValueError("Question type must be 'specific' or 'general'")
        q_types.add(q["type"])
        (single if len(q["sources"]) == 1 else multi)  # noop; just branch side effect
        if len(q["sources"]) == 1:
            single += 1
        elif len(q["sources"]) >= 2:
            multi += 1
        if len(q["a"].split()) > 15:
            raise ValueError("Answer exceeds 15-word limit")
    if q_types != {"specific", "general"}:
        raise ValueError("Must have at least one specific and one general question")
    if single != 2 or multi != 2:
        raise ValueError("Need exactly two single-source and two multi-source questions")
    person = meta["person"]
    if not isinstance(person, dict) or "entities" not in person:
        raise ValueError("'person' must be an object containing the key 'entities'")

    entities = person["entities"]
    if not isinstance(entities, list):
        raise ValueError("'person.entities' must be a list")

    seen_pairs: Set[Tuple[str, str]] = set()
    for ent in entities:
        if not (isinstance(ent, list) and len(ent) == 2):
            raise ValueError("Each entity must be a two-element list [value, type]")

        value, etype = ent
        if not (isinstance(value, str) and value.strip()):
            raise ValueError("Entity value must be a non-empty string")
        if not (isinstance(etype, str) and etype in ALLOWED_ENTITY_TYPES):
            raise ValueError(f"Invalid or disallowed entity type '{etype}'")

        if (value, etype) in seen_pairs:
            raise ValueError(f"Duplicate entity entry {ent}")
        seen_pairs.add((value, etype))


def save_cluster(idx: int, payload: Dict) -> None:
    for d in payload["documents"]:
        with open(DOCS_DIR / f"{idx:02d}_{d['id']}.json", "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    
    with open(FILES_DIR / f"cluster_{idx:02d}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    # Save evaluation info
    eval_info = {
        "cluster_id": idx,
        "cluster_risk": payload["metadata"].get("cluster_risk", ""),
        "questions_by_type": {
            "specific": [q for q in payload["metadata"]["questions"] if q["type"] == "specific"],
            "general": [q for q in payload["metadata"]["questions"] if q["type"] == "general"]
        }
    }


def update_history(payload: Dict) -> None:
    summary = payload['metadata'].get("content_summary", "")
    if summary and summary not in history_summary:
        history_summary.append(summary)

def main() -> None:
    ensure_dirs()
    api_key = os.getenv("openrouter_key")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    risk_labels = build_risk_labels(NUM_CLUSTERS)

    i = 0
    while i < NUM_CLUSTERS:
        print(f"Generating cluster: {i+1}/{NUM_CLUSTERS}")
        
        risk = risk_labels[i]
        docs_in_cluster = random.randint(*CLUSTER_SIZE_RANGE)

        if risk == "LOW":
            allowed_ent = ALLOWED_ENTITY_TYPES[5:]
            forbidden_ent = ALLOWED_ENTITY_TYPES[:5]
        elif risk == "MEDIUM":
            allowed_ent = ALLOWED_ENTITY_TYPES[1:]
            forbidden_ent = ALLOWED_ENTITY_TYPES[:1]

        else:
            allowed_ent = ALLOWED_ENTITY_TYPES
            forbidden_ent = []

        print(f"{risk}: {allowed_ent}")
        sys_prompt = SYSTEM_PROMPT.format(
            allowed_types=", ".join(allowed_ent)
        )
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            topic=TOPIC,
            cluster_index=i,
            cluster_risk=risk,
            docs_in_cluster=docs_in_cluster,
            entities_per_doc=MAX_ENTITIES_PER_DOC,
            max_link_entities_per_edge=MAX_CONNECTIONS_BETWEEN_DOC,
            used_conditions=", ".join(list(used_medical_conditions)[:10]),
            used_locations=", ".join(list(used_locations)[:10]),
            used_demographics=", ".join(list(used_demographics)[:10]),
            history_summary="; ".join(history_summary[-5:]),
            allowed_types=", ".join(allowed_ent),
            forbidden_types=", ".join(forbidden_ent)
        )
        
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
            )

            raw = resp.choices[0].message.content

            raw = extract_first_json(raw)
            print(raw)
        
            data = json.loads(raw)
            validate(data)
            save_cluster(i, data)
            update_history(data)
            
            print(f"Cluster {i+1}/{NUM_CLUSTERS} generated successfully")
            i += 1
            
        except Exception as e:
            print(f"Cluster {i+1} failed with: {e}")
            print("Retrying...")
        
        time.sleep(8) 

if __name__ == "__main__":
    main()