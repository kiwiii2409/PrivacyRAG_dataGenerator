from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# MODEL CONFIG

MODEL = "tngtech/deepseek-r1t-chimera:free"
# MODEL = "openai/gpt-oss-20b:free"
# MODEL = "deepseek/deepseek-r1-0528:free"
TEMPERATURE = 1


# DATASET
NUM_CLUSTERS = 5
CLUSTER_SIZE_RANGE = (3, 6)
MAX_ENTITIES_PER_DOC = 5
MAX_CONNECTIONS_BETWEEN_DOC = 5

RISK_SPLIT = {"HIGH": 0.40, "MEDIUM": 0.40, "LOW": 0.20}


# MISC
ALLOWED_ENTITY_TYPES = [
    "DIRECT_PII", "PHONE_NUMBER", "EMAIL", "ADDRESS", "BIRTHDATE", "AGE",
    "UNIQUE_FACT", "INDIRECT_IDENTIFIER", "MEDICAL_CONDITION", "EVENT_DATE",
    "LOCATION", "DEMOGRAPHIC", "EVENT"
]

TOPIC = "Health-Insurance Documents (Internal, Health Records, News Articles)"

# DIRECTORY
NAME = f"v12_{MODEL.split('/')[-1]}_{NUM_CLUSTERS}"
OUTPUT_ROOT = Path(f"./generated_data/{NAME}")
DOCS_DIR = OUTPUT_ROOT / "documents"
SOL_DIR = OUTPUT_ROOT / "solutions"
FILES_DIR = OUTPUT_ROOT / "files"


# diversity enforcement
history_summary = []


SYSTEM_PROMPT = """
You are a generator of synthetic health-insurance document clusters and their linkage map.
Return ONE valid JSON object only—no prose, no code fences.

Schema example
{{
  "documents": [
    {{
      "id": "doc_001",
      "content": "40-120 words of varied, realistic text; every linkable entity appears verbatim.",
      "metadata": {{"format": "email|memo|claim_form|medical_note|chat|report|news"}}
    }}
  ],
  "metadata": {{
    "category": "cluster",
    "cluster_id": "cluster_0",
    "cluster_risk": "HIGH|MEDIUM|LOW",
    "ref_id": id,
    "questions": [
      {{"q": "...?", "a": "...", "origin": ["doc_id"]}}
    ],
    "content_summary": "Short summary of the main event(s) covered by the cluster.",
    "person": {{"id": "p_0", "entities": ["...","..."]}}       ,
    "connections": {{
      "(doc_001, doc_002)": [{{"type": "EVENT_DATE", "value": "12 March 2023"}}]
    }}
  }}
}}

──────────────────────────────────────────
GENERATION RULES

1. Domain & Diversity
- Domain: health-insurance (claims, internal memos, provider reports, news).
- Clusters must cover distinct and diverse events/topics NOT present in the following earlier clusters:\n {history_summary}\n.
- Vary all proper nouns (cities, hospitals, product names, medical conditions, etc.) for realism.

2. Allowed entity types
- Only these types may appear in documents or connections: {allowed_types}.

3. Invented person
- Each cluster has exactly ONE invented person (store under metadata.person).
- All person-related entities must be among the allowed types.
- No single document, by itself, is sufficient for re-identification.

4. Risk Profile per cluster
- HIGH: ≥4 entities, ≥2 direct identifiers, 60-80 % overlap across docs.
- MEDIUM: 3-4 entities, ≤1 direct identifier, 30-50 % overlap.
- LOW: ≤3 non-direct entities, <20 % overlap.

5. Document content rules
- Each document contains 40-120 words and embeds its entities verbatim.
- Connections list only contains entities that fully appear VERBATIM in both linked docs.
- Vary connection strength within a cluster.
- Consider the question-design during content-generation, so there is suitable content to query later on

6. Question design (read carefully!)
- Produce exactly **2 questions** per cluster:
  - One “single”: answerable from **one** document.  
  - One “multi”  : requires synthesising information from **≥2** docs.
- origin: includes a list of doc_ids required to answer the question
- Questions **MUST NOT reference direct identifiers** (names, e-mails, phone numbers, etc.).
- Acceptable focus points:
  - Cluster-unique public or organisational events: policy launches, benefit changes, system outages, audit findings, regulatory deadlines.  
  - Time-based aggregates: spike in claims during “September 2023”, seasonal trends, surge of patients at “Downtown Clinic” on a specific date.  
  - Demographic summaries: leading cause of injury among “men in their 40s” mentioned in the cluster, most common diagnosis code in the dataset, etc.  
  - Financial or procedural aggregates: total reimbursement amount, average claim turnaround time, top denied CPT code, percentage of approvals.  
  - Geographic insights: region with highest claim volume, clinic experiencing unusual influx, city where a pilot programme was rolled out.  

- Unacceptable:  
  - Any question that requires revealing or inferring **direct identifiers** (names, full addresses, personal phone numbers, e-mails, SSNs).  
  - Questions tied to a single **identifiable** person, a "patient" or a "claimant" (even if the name is masked).  
  - Queries that can only be answered by cross-referencing multiple clusters or external knowledge (must stay self-contained).

- Tip: Phrase questions so they remain answerable after PII masking. For example:  
✔ “Which CPT code saw the largest month-over-month increase in September 2023?”  
✔ “What policy change triggered the spike in telehealth claims at Downtown Clinic?”  
✖ “What is John Doe's primary diagnosis?”  

- Avoid generic queries; each question must hinge on a fact unique to this cluster.
- Provide concise answers (≤15 words).

7. Output
- Deliver **only** the JSON object that fits the schema—no extra keys, no commentary.
- Validate JSON before replying; invalid output will be discarded.

DOUBLE CHECK THE OUTPUT FOR COMPLIANCE WITH THE RULES!
""".strip()


USER_PROMPT_TEMPLATE = """
Create one cluster (risk={cluster_risk}):

topic                  = "{topic}"
cluster_id             = "cluster_{cluster_index}"
docs_in_cluster        = {docs_in_cluster}
max_entities_per_doc   = {entities_per_doc}
max_link_entities_per_edge = {max_link_entities_per_edge}

Return ONLY JSON matching the system schema.
Allowed entity types: {allowed_types}
""".strip()


# HELPERS
def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    SOL_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)


def build_risk_labels(n: int) -> List[str]:
    labels = (
        ["HIGH"] * int(n * RISK_SPLIT["HIGH"]) +
        ["MEDIUM"] * int(n * RISK_SPLIT["MEDIUM"])
    )
    labels += ["LOW"] * (n - len(labels))
    random.shuffle(labels)
    return labels


def quick_validate(payload: Dict) -> None:
    # ── top-level ───────────────────────────────────────────────
    if not isinstance(payload, dict):
        raise ValueError("Top-level must be JSON object")
    if set(payload) != {"documents", "metadata"}:
        raise ValueError(
            "Top level keys must be exactly 'documents' and 'metadata'")

    docs, meta = payload["documents"], payload["metadata"]

    # ── documents ───────────────────────────────────────────────
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
        if d["metadata"]["format"] not in {
            "email", "memo", "claim_form", "medical_note",
            "chat", "report", "news"
        }:
            raise ValueError("metadata.format invalid")

    # ── metadata header ────────────────────────────────────────
    need = {"category", "cluster_id", "cluster_risk",
            "ref_id", "questions", "connections", "person"}
    if need - meta.keys():
        raise ValueError("metadata missing required keys")
    if meta["category"] not in {"cluster", "decoy", "critical"}:
        raise ValueError("metadata.category invalid")
    if meta["cluster_risk"] not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError("metadata.cluster_risk invalid")

    # ── questions: exactly 2 (1 single, 1 multi) ───────────────
    qs = meta["questions"]
    if not (isinstance(qs, list) and len(qs) == 3):
        raise ValueError("Exactly 3 questions required")
    for q in qs:
        if {"q", "a", "origin"} - q.keys():
            raise ValueError("question object malformed")

    # ── single invented person ─────────────────────────────────
    person = meta["person"]
    if not isinstance(person, dict):
        raise ValueError("metadata.person must be an object")
    if {"id", "entities"} - person.keys():
        raise ValueError("person missing 'id' or 'entities'")
    if not (isinstance(person["entities"], list) and len(person["entities"]) >= 3):
        raise ValueError("person.entities must have ≥3 items")
    # for ent in person["entities"]:
    #     if not any(ent in d["content"] for d in docs):
    #         raise ValueError(f"Person entity '{ent}' missing from all docs")

    # ── connections ────────────────────────────────────────────
    conns: Dict[str, List[Dict]] = meta["connections"]
    if not isinstance(conns, dict):
        raise ValueError("'connections' must be dict")
    for pair, ents in conns.items():
        if not (pair.startswith("(") and pair.endswith(")")):
            raise ValueError(f"Invalid connection key {pair}")
        a, b, *_ = [p.strip() for p in pair[1:-1].split(",")]
        if {a, b} - ids:
            raise ValueError("connection references unknown doc id")
        if not (isinstance(ents, list) and ents):
            raise ValueError("connections list empty")
        for e in ents:
            if {"type", "value"} - e.keys():
                raise ValueError("entity in connection missing keys")
            if e["type"] not in ALLOWED_ENTITY_TYPES:
                raise ValueError(f"invalid entity type {e['type']}")


def save_cluster(idx: int, payload: Dict) -> None:
    # documents
    for d in payload["documents"]:
        with open(DOCS_DIR / f"{idx:02d}_{d['id']}.json", "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    # connections
    with open(SOL_DIR / f"cluster_{idx:02d}_connections.json", "w", encoding="utf-8") as f:
        json.dump(payload["metadata"]["connections"],
                  f, ensure_ascii=False, indent=2)

    # full json
    with open(FILES_DIR / f"cluster_{idx:02d}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
        print(f"cluster: {i}/{NUM_CLUSTERS}")
        risk = risk_labels[i]
        docs_in_cluster = random.randint(*CLUSTER_SIZE_RANGE)

        sys_prompt = SYSTEM_PROMPT.format(
            allowed_types=", ".join(ALLOWED_ENTITY_TYPES),
            history_summary="; ".join(
                history_summary) if history_summary else "none"
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            topic=TOPIC,
            cluster_index=i,
            cluster_risk=risk,
            docs_in_cluster=docs_in_cluster,
            entities_per_doc=MAX_ENTITIES_PER_DOC,
            max_link_entities_per_edge=MAX_CONNECTIONS_BETWEEN_DOC,
            allowed_types=", ".join(ALLOWED_ENTITY_TYPES),
        )

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        try:
            data = json.loads(raw)
            print(data)
        except Exception as e:
            output = raw[:400].replace("\n", "\\n")
            raise ValueError(
                f"Model did not return valid JSON: {e}\nOutput: {output}")

        try:
            quick_validate(data)
            save_cluster(i, data)
            update_history(data)

            i += 1
        except Exception as e:
            print(
                f"ERROR: Cluster {i} didn't fulfill the requirements and was skipped. Failed with {e}")
            print("Retrying...")

        # stupid rate limits
        time.sleep(3)


if __name__ == "__main__":
    main()
