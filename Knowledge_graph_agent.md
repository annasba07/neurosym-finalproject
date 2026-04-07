# CareTrace — Knowledge Retrieval Agent

## Overview

The Knowledge Retrieval Agent is the symbolic backbone of CareTrace. It grounds free-text caregiver language into canonical SNOMED concepts, traverses the IS_A hierarchy to generalize findings, retrieves authoritative red flags, and checks medication dosing and contraindications — all from a scoped Neo4j knowledge graph.

All other agents interact with this module through `KnowledgeRetrievalAgent` in `agent.py`. Nothing else should query Neo4j directly.

---

## File Structure

```
knowledge_graph_agent/
├── __init__.py       # Public exports
├── schema.py         # Node labels, relationship types, property keys, enums
├── loader.py         # Populates Neo4j with concepts, medications, red flags, edges
├── queries.py        # All Cypher queries — returns clean Python dicts
└── agent.py          # Public interface used by LangGraph orchestration
```

---

## Knowledge Graph Design

### Node Types

| Label | Description | Example |
|-------|-------------|---------|
| `Concept` | SNOMED clinical concept | `Fever (finding)` · SCTID `386661006` |
| `Medication` | Drug with full dosing metadata | `ibuprofen` · `acetaminophen` |
| `RedFlag` | Hard escalation trigger with disposition | `RF_004` · Neck stiffness with fever |

### Relationship Types

| Relationship | Direction | Meaning |
|---|---|---|
| `IS_A` | child → parent | SNOMED hierarchy traversal |
| `ASSOCIATED_WITH` | Concept → RedFlag | Triggers escalation rule |
| `TREATED_BY` | Concept → Medication | What treats this finding |
| `CONTRAINDICATED_IN` | Medication → Concept | Unsafe under this condition |

### Scoped SNOMED Concepts

The graph covers exactly the pediatric febrile illness bundle required by the project scope. Concepts outside this bundle are not represented — the agent defaults to requesting clinician evaluation rather than fabricating certainty.

**Fever cluster**
- `386661006` — Fever (finding): `fever`, `pyrexia`, `high temperature`, `burning up`, `hot`, `febrile`
- `248425001` — Febrile convulsion (disorder): `febrile seizure`, `shaking with fever`
- `271807003` — Rash (finding): `rash`, `spots`, `blotches`
- `271757001` — Petechiae (finding): `pin-point spots`, `non-blanching rash`

**GI cluster**
- `422587007` — Nausea (finding): `nausea`, `feeling sick`, `queasy`
- `422400008` — Vomiting (disorder): `vomiting`, `throwing up`, `puking`
- `62315008` — Diarrhea (finding): `loose stools`, `watery stool`, `runny poop`
- `21522001` — Abdominal pain (finding): `stomach pain`, `belly ache`, `tummy hurts`

**Dehydration cluster**
- `34095006` — Dehydration (disorder): `dehydration`, `dehydrated`, `not drinking`
- `28442001` — Decreased urine output (finding): `not urinating`, `no wet diapers`, `hasn't peed`
- `1290011000` — Dry mucous membranes (finding): `dry mouth`, `dry lips`, `parched`
- `271594007` — Sunken eyes (finding): `sunken eyes`, `eyes look hollow`

**Neurological / severity**
- `40917007` — Lethargy (finding): `lethargic`, `won't wake up`, `listless`, `unresponsive`
- `162397003` — Neck stiffness (finding): `stiff neck`, `can't bend neck`
- `230145002` — Difficulty breathing (finding): `trouble breathing`, `breathing fast`, `respiratory distress`

### IS_A Hierarchy

```
Febrile convulsion  ──IS_A──►  Fever
Petechiae           ──IS_A──►  Rash
Nausea              ──IS_A──►  Vomiting
Decreased urine     ──IS_A──►  Dehydration
Dry mucous memb.    ──IS_A──►  Dehydration
Sunken eyes         ──IS_A──►  Dehydration
```

Hierarchy traversal means a patient presenting with "sunken eyes" automatically inherits the Dehydration concept — and its associated red flag (RF_007) — without the rule needing to enumerate every dehydration sign explicitly.

---

## Red Flag Rules

| Rule ID | Description | Disposition | Triggered By |
|---------|-------------|-------------|--------------|
| RF_001 | Fever in infant under 3 months | ED_NOW | Fever |
| RF_002 | Febrile seizure | ED_NOW | Febrile convulsion |
| RF_003 | Non-blanching petechial rash with fever | ED_NOW | Petechiae |
| RF_004 | Neck stiffness with fever | ED_NOW | Neck stiffness |
| RF_005 | Lethargy or inability to rouse | ED_NOW | Lethargy |
| RF_006 | Respiratory distress | ED_NOW | Difficulty breathing |
| RF_007 | Severe dehydration: no urine > 8 hours, sunken eyes, dry mucous membranes | ED_NOW | Dehydration |
| RF_008 | Fever persisting > 5 days | URGENT_CARE | — |
| RF_009 | Vomiting preventing oral rehydration | URGENT_CARE | Vomiting |
| RF_010 | Moderate dehydration: no urine > 6 hours | URGENT_CARE | Decreased urine output |

> **Note:** RF_001 (fever in infant < 3 months) fires on any fever concept. Age gating must be applied by the Safety Logic Agent using patient context — the KG does not enforce age independently.

Disposition priority: `ED_NOW` > `URGENT_CARE` > `HOME`. The agent always surfaces the highest-severity disposition when multiple red flags fire.

---

## Medications

| Field | Acetaminophen | Ibuprofen |
|-------|--------------|-----------|
| Formulation | oral suspension 160mg/5ml | oral suspension 100mg/5ml |
| Dose | 15 mg/kg | 10 mg/kg |
| Max dose | 1000 mg | 600 mg |
| Max daily doses | 5 | 4 |
| Min interval | 4 hours | 6 hours |
| Min age | 2 months | 6 months |
| Min weight | 3.0 kg | 5.0 kg |
| Source | AAP 2023 | AAP 2023 |

**Contraindication:** Ibuprofen is contraindicated under `Dehydration (disorder)` (SCTID `34095006`). Because of the IS_A hierarchy, this contraindication is also triggered if the patient presents with decreased urine output, dry mucous membranes, or sunken eyes — as these all inherit the Dehydration concept.

---

## Public API

All methods are on `KnowledgeRetrievalAgent`. Use it as a context manager:

```python
from knowledge_graph_agent.agent import KnowledgeRetrievalAgent

with KnowledgeRetrievalAgent() as agent:
    result = agent.ground_symptom("burning up")
```

### `ground_symptom(text_mention: str) → dict`

Maps a single free-text mention to SNOMED concept(s) via synonym matching. Walks the IS_A chain to return ancestors automatically.

```python
agent.ground_symptom("burning up")
# → { "mention": "burning up", "concepts": [...], "all_sctids": ["386661006"], "grounded": True }
```

Always use `result["all_sctids"]` for downstream calls — it includes ancestors.

### `ground_symptoms_batch(text_mentions: list[str]) → dict`

Grounds multiple mentions at once and returns a merged, deduplicated `all_sctids` list. This is the main entry point during a triage turn.

```python
agent.ground_symptoms_batch(["fever", "not urinating", "stiff neck"])
# → { "results": {...}, "all_sctids": [...], "ungrounded": [...] }
```

### `get_red_flags(sctids: list[str]) → dict`

Returns all red flags triggered by the given concept IDs (including ancestors). Always pass the full expanded list from `ground_symptoms_batch`.

```python
agent.get_red_flags(grounding["all_sctids"])
# → { "red_flags": [...], "highest_disposition": "ED_NOW", "has_red_flags": True }
```

### `get_dosing(medication_name: str, weight_kg: float) → dict`

Returns authoritative weight-based dosing with computed dose_mg and dose_ml. Source-attributed to AAP 2023.

```python
agent.get_dosing("ibuprofen", 14.0)
# → { "dose_mg": 140.0, "dose_ml": 7.0, "formulation": "oral suspension 100mg/5ml", ... }
```

### `get_contraindications(medication_name: str, sctids: list[str]) → dict`

Checks whether a medication is contraindicated given the patient's active concepts.

```python
agent.get_contraindications("ibuprofen", grounding["all_sctids"])
# → { "contraindicated": True, "reasons": [{ "contraindicated_in": "Dehydration (disorder)", ... }] }
```

### `get_safe_medications(sctids, weight_kg, age_months) → list[dict]`

Convenience method that combines dosing + contraindication checks for all scoped medications. Returns a `safe` boolean per medication.

### `explain_grounding(text_mention: str) → str`

Returns a human-readable audit string for the Explanation Agent. Shows the IS_A chain explicitly.

```
"burning up" grounded to:
  • Fever (finding) (SCTID: 386661006)
```

---

## Scenario Coverage

### Scenario 1 — Moderate fever, child on amoxicillin (Home management)

Key KG operations:
1. `ground_symptoms_batch(["fever", "vomiting", "fatigue"])` → fever + vomiting grounded
2. `ground_symptoms_batch(["not urinating"])` absent → no dehydration signal
3. `get_red_flags(all_sctids)` → RF_009 may fire (vomiting), but no ED_NOW triggers
4. `get_contraindications("ibuprofen", all_sctids)` → safe if no dehydration present
5. `get_dosing("ibuprofen", weight_kg)` → precise dose returned with AAP source

### Scenario 2 — High fever, barely responding, no urination (ER escalation)

Key KG operations:
1. `ground_symptoms_batch(["fever", "lethargic", "not drinking"])` → lethargy grounded → IS_A chain checked
2. `ground_symptoms_batch(["no urination"])` → decreased urine output → IS_A Dehydration
3. `get_red_flags(all_sctids)` → RF_005 (lethargy → ED_NOW) + RF_007 (dehydration → ED_NOW)
4. `highest_disposition` → `ED_NOW`
5. `get_contraindications("ibuprofen", all_sctids)` → contraindicated (dehydration present)

---

## Known Limitations

- **RF_001 age gating:** The KG fires RF_001 on any fever regardless of patient age. The Safety Logic Agent must filter this using the patient's age in months before acting on the disposition.
- **Numeric fever grounding:** Strings like `"fever 101.8"` do not match synonyms. The Interpretation Agent must separate temperature values from symptom labels before calling `ground_symptoms_batch`.
- **"Fatigue" not grounded:** The synonym list does not include `fatigue` as a mapped term. Consider adding it as a synonym for `Lethargy (finding)` or as a separate concept if it needs to trigger rules independently.
- **Local context:** The schema defines a `LocalContext` node label but no nodes are loaded. This is a stub for epidemiological priors (e.g., "RSV season"). Scenario 2's school virus context is currently handled by the Explanation Agent as a probabilistic prior, not by the KG.
- **Closed-world assumption:** The agent only covers the scoped pediatric febrile illness bundle. Queries for concepts outside this scope will return `grounded: False`. This is by design — the system defaults to requesting clinician evaluation rather than fabricating an answer.