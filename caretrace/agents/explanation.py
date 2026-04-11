"""Explanation agent — verbalizes the structured decision for caregivers.

The LLM receives the full decision object (disposition, rule trace,
positives/negatives, red flags) and turns it into natural, empathetic,
caregiver-facing language. It does NOT make clinical decisions.
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from caretrace.state import ClinicalState, FACT_QUESTIONS


# Human-readable labels for the caregiver-facing disposition line.
DISPOSITION_LABELS = {
    "er_now":       "GO TO THE EMERGENCY ROOM NOW",
    "urgent_eval":  "SEE A DOCTOR URGENTLY (within the next few hours)",
    "home_monitor": "SAFE TO MANAGE AT HOME WITH MONITORING",
}


EXPLAIN_SYSTEM_PROMPT = """\
You are a caring pediatric nurse providing triage guidance to a worried parent.

You will receive a structured clinical decision. Your job is to turn it into
clear, empathetic, actionable guidance. You must:

1. State the recommendation clearly up front (ER now / urgent care / safe to monitor).
2. Explain WHY using the key findings and red flags provided — be specific.
3. List what to watch for that would mean going to the ER immediately.
4. If home management: give a concise overnight care plan (fluids, rest, fever control).
5. Keep it short — this is a triage summary, not a medical essay.

Rules:
- NEVER contradict the disposition — if the system says ER, you say ER.
- Use simple language — the caregiver is stressed and may not be medically trained.
- Be warm but direct — don't hedge on safety-critical advice.
- End with brief reassurance appropriate to the situation.
"""

FOLLOWUP_SYSTEM_PROMPT = """\
You are a caring pediatric triage nurse collecting information from a worried parent.

The system still needs information to complete the assessment. Ask for the
missing information naturally and warmly. You may combine 1-2 questions if
they flow naturally together, but don't overwhelm with too many questions at once.

Prioritize safety-critical questions first (alertness, breathing).
"""


def _get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def explain(state: ClinicalState) -> dict:
    """LangGraph node: generate the caregiver-facing explanation."""
    disposition = state.get("disposition")
    if disposition is None:
        return {}

    decision_summary = _build_decision_summary(state)

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=EXPLAIN_SYSTEM_PROMPT),
        HumanMessage(content=decision_summary),
    ])

    return {
        "explanation": response.content,
        "messages":    [AIMessage(content=response.content)],
        "is_complete": True,
    }


def ask_followup(state: ClinicalState) -> dict:
    """LangGraph node: ask for missing facts when disposition is undecided."""
    missing = state.get("missing_required", [])
    if not missing:
        return {}

    # Take the top two missing facts by priority order
    questions = []
    for key in missing[:2]:
        q = FACT_QUESTIONS.get(key)
        if q:
            questions.append(q)

    if not questions:
        return {}

    known_summary = _build_known_summary(state)

    llm = _get_llm()
    prompt = (
        f"What we know so far:\n{known_summary}\n\n"
        f"We still need to ask:\n"
        + "\n".join(f"- {q}" for q in questions)
        + "\n\nGenerate a warm, natural follow-up message combining these questions."
    )
    response = llm.invoke([
        SystemMessage(content=FOLLOWUP_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {"messages": [AIMessage(content=response.content)]}


# ── Summary builders ──────────────────────────────────────────────────────────

def _build_decision_summary(state: ClinicalState) -> str:
    lines: list[str] = []
    disposition = state.get("disposition", "")
    label = DISPOSITION_LABELS.get(disposition, disposition.upper())
    lines.append(f"DISPOSITION: {label}")
    lines.append("")

    positives = state.get("key_positives", [])
    negatives = state.get("key_negatives", [])
    if positives:
        lines.append("KEY CONCERNS:")
        for p in positives:
            lines.append(f"  - {p}")
    if negatives:
        lines.append("")
        lines.append("REASSURING FINDINGS:")
        for n in negatives:
            lines.append(f"  - {n}")

    # Red-flag provenance — show which backend flagged what
    red_flags = state.get("kg_red_flags", [])
    if red_flags:
        lines.append("")
        lines.append("RED FLAGS TRIGGERED:")
        for f in red_flags:
            src = f.get("source", "kg")
            lines.append(f"  - [{src}] {f['description']} ({f['disposition']})")

    # Rules trace (Alex's + KG merged) — useful context for the LLM
    rules = state.get("rules_triggered", [])
    if rules:
        lines.append("")
        lines.append("RULE TRACE:")
        for r in rules:
            lines.append(f"  - {r}")

    # Raw clinical context
    facts_bits = []
    if state.get("age_months") is not None:
        age = state["age_months"]
        facts_bits.append(
            f"age: {age // 12}y {age % 12}m" if age >= 24 else f"age: {age} months"
        )
    if state.get("temperature_f") is not None:
        facts_bits.append(f"temp: {state['temperature_f']}°F")
    if state.get("current_medication"):
        facts_bits.append(f"medication: {state['current_medication']}")
    if facts_bits:
        lines.append("")
        lines.append("CONTEXT: " + ", ".join(facts_bits))

    return "\n".join(lines)


def _build_known_summary(state: ClinicalState) -> str:
    """Friendly summary of what we already know, for the follow-up prompt."""
    bits: list[str] = []
    if state.get("age_months") is not None:
        age = state["age_months"]
        bits.append(f"Age: {age // 12} years old" if age >= 24 else f"Age: {age} months old")
    if state.get("temperature_f") is not None:
        bits.append(f"Temperature: {state['temperature_f']}°F")

    facts = state.get("facts", {}) or {}
    if "fever" in facts:
        bits.append(f"Fever: {facts['fever']}")
    if "alert" in facts:
        bits.append(f"Alertness: {facts['alert']}")
    if "intake" in facts:
        bits.append(f"Drinking: {facts['intake']}")
    if "urination" in facts:
        bits.append(f"Urination: {facts['urination']}")
    if "vomiting" in facts:
        bits.append(f"Vomiting: {facts['vomiting']}")
    if "breathing" in facts:
        bits.append(f"Breathing: {facts['breathing']}")
    if state.get("current_medication"):
        bits.append(f"Medication: {state['current_medication']}")

    return "\n".join(bits) if bits else "Limited information so far."
