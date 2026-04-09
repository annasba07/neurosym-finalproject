# CareTrace — Knowledge Retrieval Agent

Grounds caregiver language into SNOMED concepts, retrieves red flags, and checks medication dosing/contraindications from Neo4j.

**No other agent should query Neo4j directly.**

---

## Setup

```bash
pip install neo4j python-dotenv
python -m knowledge_graph_agent.loader
```

`.env` requires `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

---

## State

The KG agent reads and writes to the shared LangGraph state:

```json
{
  "facts": {
    "fever": "yes | no | unknown",
    "alert": "normal | reduced",
    "vomiting": "none | once | repeated | unknown",
    "intake": "normal | reduced | none",
    "urination": "normal | unknown | none",
    "age_months": 72,
    "weight_kg": 20.0
  },
  "kg_concepts": ["386661006", "422400008"],
  "derived_flags": [
    {
      "rule_id": "RF_009",
      "description": "Vomiting preventing oral rehydration",
      "disposition": "URGENT_CARE",
      "triggered_by": "Vomiting (disorder)"
    }
  ]
}
```

- **Reads:** `facts`
- **Writes:** `facts` (preserved + KG-corrected), `kg_concepts`, `derived_flags`

---

## API

```python
from knowledge_graph_agent.agent import KnowledgeRetrievalAgent

with KnowledgeRetrievalAgent() as agent:
    result = agent.ground_symptom("burning up")
```

Always pass `all_sctids` (includes ancestors) to any method that takes `sctids`.

---

### `ground_symptom(text_mention: str)`
- **In:** single symptom string
- **Out:** `all_sctids`, `grounded`

### `ground_symptoms_batch(text_mentions: list[str])`
- **In:** list of symptom strings
- **Out:** `all_sctids`, `ungrounded`

### `get_red_flags(sctids: list[str], age_months: int)`
- **In:** expanded sctid list, patient age in months
- **Out:** `red_flags`, `highest_disposition`, `has_red_flags`
- **Note:** `age_months` is required — raises `ValueError` if `None`. RF_001 suppressed when `age_months >= 3`.

### `get_dosing(medication_name: str, weight_kg: float)`
- **In:** `"acetaminophen"` or `"ibuprofen"`, child weight in kg
- **Out:** `dose_mg`, `dose_ml`, `formulation`, `max_dose_mg`, `min_interval_hours`, `max_daily_doses`, `min_age_months`, `source`, `safe_for_weight`, `found`

### `get_contraindications(medication_name: str, sctids: list[str])`
- **In:** medication name, expanded sctid list
- **Out:** `contraindicated`, `reasons`

### `get_safe_medications(sctids: list[str], weight_kg: float, age_months: int)`
- **In:** expanded sctid list, weight, age
- **Out:** list of medications each with `safe`, `age_eligible`, `weight_eligible`, `contraindicated`, `contraindication_reasons`

### `explain_grounding(text_mention: str)`
- **In:** single symptom string
- **Out:** human-readable IS_A chain string for the Explanation Agent

---

## Known Limitations

- `tired`, `wiped out`, `fatigue` — not grounded; Interpretation Agent must normalize to `lethargic`
- Numeric fever strings (e.g. `"fever 101.8"`) — not grounded; strip values before calling
- RF_009 fires on vomiting alone — Safety Logic Agent must confirm oral intake is blocked
- `LocalContext` node stubbed but not loaded