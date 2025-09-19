from __future__ import annotations
import argparse, json, os, time, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()




def call_with_retry(
        func: Callable, *args,
        max_tries: int = 3,
        retry_sleep: float = 1.5,
        **kwargs):
    """Minimal retry wrapper with exponential back-off (no logging)."""
    for attempt in range(max_tries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt == max_tries - 1:
                raise
            time.sleep(retry_sleep * (attempt + 1))

ENTITY_LIST = [
    "NAME","PATIENT_ID","PROCESS_ID","PHONE_NUMBER","EMAIL","ADDRESS","BIRTHDATE",
    "AGE","UNIQUE_FACT","INDIRECT_IDENTIFIER","MEDICAL_CONDITION","EVENT_DATE",
    "LOCATION","DEMOGRAPHIC","EVENT", "PROVIDER"
]
PROMPT_HEADER = f"""
You are a data-privacy analyst. Each cluster contains documents that together may reveal the identity of a single patient or claimant.  

Your task: Return **one JSON object** with exactly these keys:

2. "links":  

   - Identify chains of **two or three documents** that are clearly connected and, when combined, significantly increase the re-identification risk of the patient.  
   - Identification can happen via direct identifiers but also indirect identifiers that become dangerous when accumulated

   - Output format:  
     ```json
     "links": [["doc1","doc2","doc3"], ["doc2","doc3"]]
     ```


Final Output Format:
--------------------
```json
{{
  "links": [...]
}}
```

Return **only** valid JSON exactly in that shape - no extra keys, no commentary.

CLUSTER:
"""
PROMPT_HEADER_orig = f"""
You are a data-privacy analyst. Each cluster contains documents that together may reveal the identity of a single patient or claimant.  

Your task: Return **one JSON object** with exactly these keys:

1. "entities":  
    - Extract an entity only if it belongs to, describes, or can help re-identify the patient. 
    - Use only these types: {", ".join(ENTITY_LIST)}. 
    - If a phone / email/ etc. clearly belongs to a doctor or other provider, SKIP it, even if it appears in the text verbatim. 
    - For each entity, consider how it might be used for re-identification. Only add it if you can find a realistic and reasonable scenario 
    - Standardise obvious variants (e.g., “19.12.2001” → “2001-12-19”, "32-year-old" → "32", "... near/ close to [named_location]" → [name_location]). 
    - Extract entities on a per-document basis, but consider other documents as context to decide which entities to include
    - Only if a doctor or providers name is listed, extract it!
   - Output format:  
     ```json
     "entities": {{
       "document_id": ["entity1", "entity2"]
     }}
     ```

2. "links":  
   - Identify chains of **two or three documents** that are clearly connected by entities and, when combined, significantly increase the re-identification risk of the patient.  
   - only chain documents that actually belong to 

   - Output format:  
     ```json
     "links": [["doc1","doc2","doc3"], ["doc2","doc3"]]
     ```

3. "redaction":  
   - Provide the **minimal set of entities** that must be redacted so re-identification is not possible, even if documents are combined.  
   - Preserve cross-document utility where possible.  
   - Output format:  
     ```json
     "redaction": ["entity1", "entity2"]
     ```

Final Output Format:
--------------------
```json
{{
  "entities": {{...}},
  "links": [...],
  "redaction": [...]
}}
```

Return **only** valid JSON exactly in that shape - no extra keys, no commentary.

CLUSTER:
"""

# --------------------------------------------------------------------------- helpers
def build_prompt(cluster: Dict[str, Any]) -> str:
    return PROMPT_HEADER + json.dumps(cluster, ensure_ascii=False, indent=2)

def llm_call(client: OpenAI, model: str, prompt: str) -> Dict[str, Any]:
    """Single OpenAI chat call that must return JSON."""
    rsp = client.chat.completions.create(
        model=model,
        temperature=0.01,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ],
    )
    return json.loads(rsp.choices[0].message.content.strip())

def merge_clusters(cluster_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-cluster JSONs into the global gold file."""
    merged_entities = {}
    links = set()
    redaction = []
    for ans in cluster_answers:
        # merged_entities.update(ans["entities"])
        for x in ans["links"]:
            if len(x) ==3:
                links.add(tuple(sorted((x[0], x[1], x[2]))))
            elif len(x) == 2:
                links.add(tuple(sorted((x[0], x[1]))))

        # redaction.extend(ans["redaction"])

    return {
        "entities": merged_entities,
        "links":    sorted(list(links)),
        "redaction": redaction
    }


EXAMPLE = "v2_d0_10_deepseek-r1t-chimera:free_0"
CLUSTER_FOLDER = f"./generated_data/{EXAMPLE}/files"            
OUTPUT_FILE    = f"./golden_answers/{EXAMPLE}_sol.json"    
MODEL_ID       = "openai/gpt-oss-120b"   
OPENROUTER_KEY =  os.getenv("openrouter_key")

def main() -> None:
    cluster_dir = Path(CLUSTER_FOLDER)
    if not cluster_dir.is_dir():
        sys.exit(f"ERROR: folder '{cluster_dir}' not found")

    json_files = sorted(cluster_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"ERROR: no *.json files in '{cluster_dir}'")

    client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
    answers = []

    for jf in json_files:
        print(f"• processing {jf.name}", file=sys.stderr)
        cluster_obj = json.loads(jf.read_text(encoding="utf-8"))
        prompt      = build_prompt(cluster_obj.get("documents"))
        result      = call_with_retry(llm_call, client, MODEL_ID, prompt)
        print(result)
        answers.append(result)
    
    merged = merge_clusters(answers)

    Path(OUTPUT_FILE).write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"Golden evaluation saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()