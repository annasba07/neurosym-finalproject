# CareTrace — Neurosym Final Project

A multi-agent pediatric triage system that converts caregiver free-text into structured clinical facts, then routes to knowledge retrieval and safety logic agents.

## Overview

CareTrace uses a LangGraph pipeline to conduct a structured clinical interview with caregivers, extracting symptoms and vital information one turn at a time — without ever generating medical advice.

```
Caregiver message
      │
      ▼
Interpretation Agent   ←── this repo
  (LLM extraction)
      │
      ├─ incomplete → ask follow-up question → wait for next turn
      │
      └─ complete ──► Knowledge Retrieval Agent (Neo4j KG)
                              │
                              ▼
                      Knowledge Graph Agent 
```

## Agent Architecture

### Interpretation Agent (`interpretation_agent.ipynb`)

**Responsibilities:**
1. Named entity extraction — symptoms, duration, hydration signals, medications
2. Missing-field detection — identifies which required fields are still unknown
3. Follow-up question generation — exactly **one** targeted question per turn

**Hard constraints:**
- One question per turn (no information bombardment)
- Must NOT generate clinical advice, diagnosis, or disposition
- Only updates a field when the caregiver message provides new information

**Output to downstream agents:**
```python
{
    "case_facts": dict,          # clinical fields → passed to KG retrieval
    "is_complete": bool,         # True → KG+rules, False → ask follow-up
    "follow_up_question": str,   # shown to caregiver if is_complete=False
}
```

## Clinical Fields Extracted

| Field | Type | Values |
|---|---|---|
| `fever` | str | `"yes"` / `"no"` |
| `vomiting` | str | `"none"` / `"once"` / `"repeated"` |
| `alert` | str | `"normal"` / `"reduced"` |
| `intake` | str | `"normal"` / `"reduced"` / `"none"` |
| `urination` | str | `"normal"` / `"unknown"` / `"none"` |


## Setup

### Prerequisites
- Python 3.10+
- Groq API key (or swap for Claude / GPT-4o)

### Installation

```bash
pip install langgraph langchain-groq pydantic python-dotenv tenacity
```

### Environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

## LLM Configuration

The agent currently uses `llama-3.3-70b-versatile` via Groq. To swap models, replace the `ChatGroq` block in Cell 6 with your preferred provider (Claude, GPT-4o, etc.).

## Adding or Modifying Clinical Fields

To add a new clinical field, update all of the following in `interpretation_agent.ipynb`:

1. **`ClinicalState`** (Cell 9) — add the field to the persistent graph state
2. **`ExtractionResult`** (Cell 11) — add the field with a Pydantic `Field` description so the LLM knows what to extract
3. **`EXTRACTION_SYSTEM_PROMPT`** (Cell 13) — describe extraction rules for the field
4. **`REQUIRED_FIELDS`** (Cell 15) — add the field and its follow-up question (if required)
5. **`clinical_fields` list** (Cell 28) — include it in `case_facts` passed to downstream agents

## Evaluation

See `evaluation.ipynb` for test cases and constraint verification:
- Single-question constraint (no information bombardment)
- No clinical advice in output
- Completeness detection (`is_complete` flag)

Golden dataset: `golden_dataset.json`

## Project Structure

```
neurosym-finalproject/
├── interpretation_agent.ipynb   # main agent implementation
├── evaluation.ipynb             # test cases and evaluation
├── golden_dataset.json          # ground-truth cases for evaluation
├── .env                         # API keys (not committed)
└── README.md
```
