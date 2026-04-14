from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Any
import sys
import os
from pathlib import Path

# Make sure caretrace package is importable from repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from langchain_core.messages import HumanMessage, AIMessage
from caretrace.graph import create_app
from caretrace.state import initial_state

app = FastAPI(title="CareTrace Demo")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "demo-thread"
    # Serialized state fields we round-trip from the client
    facts: dict = {}
    age_months: Optional[float] = None
    temperature_f: Optional[float] = None
    fever_duration_days: Optional[float] = None
    current_medication: Optional[str] = None
    raw_symptoms: list = []
    grounded_concepts: list = []
    all_sctids: list = []
    kg_red_flags: list = []
    observation_predicates: list = []
    concern_predicates: list = []
    decision: Optional[str] = None
    rules_triggered: list = []
    disposition: Optional[str] = None
    missing_required: list = []
    follow_up_question: Optional[str] = None
    explanation: Optional[str] = None
    key_positives: list = []
    key_negatives: list = []
    go_now_thresholds: list = []
    overnight_plan: list = []
    is_complete: bool = False
    phase: Optional[str] = None
    turn: int = 0
    kg_backend: Optional[str] = None
    weight_kg: Optional[float] = None
    medication_last_dose: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    disposition: Optional[str]
    phase: Optional[str]
    facts: dict
    rules_triggered: list
    kg_red_flags: list
    observation_predicates: list
    concern_predicates: list
    missing_required: list
    follow_up_question: Optional[str]
    is_complete: bool
    age_months: Optional[float]
    temperature_f: Optional[float]
    # Pass full state back for next round-trip
    state_snapshot: dict


@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Rebuild state from round-tripped client data
    s = initial_state()
    s["messages"] = []
    s["facts"] = req.facts or {}
    s["age_months"] = req.age_months
    s["temperature_f"] = req.temperature_f
    s["fever_duration_days"] = req.fever_duration_days
    s["current_medication"] = req.current_medication
    s["raw_symptoms"] = req.raw_symptoms or []
    s["grounded_concepts"] = req.grounded_concepts or []
    s["all_sctids"] = req.all_sctids or []
    s["kg_red_flags"] = req.kg_red_flags or []
    s["observation_predicates"] = req.observation_predicates or []
    s["concern_predicates"] = req.concern_predicates or []
    s["decision"] = req.decision
    s["rules_triggered"] = req.rules_triggered or []
    s["disposition"] = req.disposition
    s["missing_required"] = req.missing_required or []
    s["follow_up_question"] = req.follow_up_question
    s["explanation"] = req.explanation
    s["key_positives"] = req.key_positives or []
    s["key_negatives"] = req.key_negatives or []
    s["go_now_thresholds"] = req.go_now_thresholds or []
    s["overnight_plan"] = req.overnight_plan or []
    s["is_complete"] = req.is_complete
    s["phase"] = req.phase
    s["turn"] = req.turn
    s["kg_backend"] = req.kg_backend
    s["weight_kg"] = req.weight_kg
    s["medication_last_dose"] = req.medication_last_dose

    # Add the new user message
    s["messages"] = [HumanMessage(content=req.message)]

    # Run the LangGraph pipeline
    caretrace_app, config = create_app(thread_id=req.thread_id)
    result = caretrace_app.invoke(s, config)

    # Extract the latest AI reply
    reply = ""
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage):
            reply = m.content
            break

    # Build a clean state snapshot for the next round-trip
    # (exclude messages — we don't need to round-trip the full history)
    snapshot = {k: v for k, v in result.items() if k != "messages"}

    return ChatResponse(
        reply=reply,
        disposition=result.get("disposition"),
        phase=result.get("phase"),
        facts=result.get("facts", {}),
        rules_triggered=result.get("rules_triggered", []),
        kg_red_flags=result.get("kg_red_flags", []),
        observation_predicates=result.get("observation_predicates", []),
        concern_predicates=result.get("concern_predicates", []),
        missing_required=result.get("missing_required", []),
        follow_up_question=result.get("follow_up_question"),
        is_complete=result.get("is_complete", False),
        age_months=result.get("age_months"),
        temperature_f=result.get("temperature_f"),
        state_snapshot=snapshot,
    )