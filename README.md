# CareTrace – Rules Agent & Evaluation

This repo contains the **rule-based triage decision engine** for the CareTrace project, along with LLM comparison scripts and demo notebook.

---

## Repository Structure

```
.
├── src/
│   └── rules.py                # Core rules agent 
│
├── llm_openrouter.py           # LLM call (OpenRouter)
├── test_compare_models.py      # LLM vs rules comparison script
├── test_rules.py               # Unit tests for rules agent
├── CareTrace_Rules_Demo.ipynb  # Demo notebook (used for presentation)
│
├── .gitignore
└── README.md
```

---

## Rules Agent 

### Entry point:

```python
from src.rules import rules_agent
```

### Input:

```python
state = {
    "facts": {
        "fever": "yes | no | unknown",
        "alert": "normal | reduced",
        "vomiting": "none | once | repeated | unknown",
        "intake": "normal | reduced | none",
        "urination": "normal | unknown | none"
    }
}
```

### Output:

```python
{
    "decision": "home_monitor | urgent_eval | er_now | unsupported",
    "observation_predicates": [...],
    "concern_predicates": [...],
    "rules_triggered": [...]
}
```

---

## Rules Architecture

The rules agent follows a layered structure:

```
facts
  ↓
observation_predicates
  ↓
concern_predicates
  ↓
decision
```

### Example:

```
poor_intake + fever_present
→ dehydration_concern
→ urgent_eval
```

## Running Tests

### Rules only:

```bash
python test_rules.py
```

### LLM vs Rules:

```bash
python test_compare_models.py
```

---

## Demo Notebook

```
CareTrace_Rules_Demo.ipynb
```

For:
- scenario walkthroughs  
- LLM vs rules comparison  
- presentation  
