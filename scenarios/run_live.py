"""Scripted end-to-end run of both clinical scenarios through the live LangGraph.

Uses the real Groq LLM for interpretation + explanation, Alex's plain-Python
rules engine, and the KG adapter (Neo4j if reachable, dict fallback otherwise).

Run with: python -m scenarios.run_live
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from caretrace.graph import create_app
from caretrace.state import initial_state


def _print_bubble(role: str, text: str) -> None:
    label = "USER" if role == "user" else "CARETRACE"
    print(f"\n[{label}]")
    for line in text.splitlines():
        print(f"  {line}")


def _print_state_summary(state: dict) -> None:
    print("\n  --- state snapshot ---")
    print(f"    disposition : {state.get('disposition') or 'undecided'}")
    print(f"    phase       : {state.get('phase')}")
    print(f"    kg backend  : {state.get('kg_backend')}")

    facts = state.get("facts") or {}
    if facts:
        print(f"    facts       : {facts}")

    missing = state.get("missing_required", [])
    if missing:
        print(f"    missing     : {missing}")

    obs = state.get("observation_predicates", [])
    if obs:
        print(f"    observations: {obs}")

    concerns = state.get("concern_predicates", [])
    if concerns:
        print(f"    concerns    : {concerns}")

    rules = state.get("rules_triggered", [])
    if rules:
        print("    rule trace  :")
        for r in rules:
            print(f"      - {r}")

    red_flags = state.get("kg_red_flags", [])
    if red_flags:
        print("    red flags   :")
        for f in red_flags:
            src = f.get("source", "kg")
            print(f"      [{src}] {f['rule_id']} → {f['disposition']}")

    grounded = state.get("grounded_concepts", [])
    if grounded:
        mentions = [c.get("mention", "?") for c in grounded]
        print(f"    grounded    : {mentions}")

    print("  ----------------------")


def run_scenario(title: str, messages: list[str], thread_id: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    app, config = create_app(thread_id=thread_id)
    state = initial_state()

    for msg in messages:
        _print_bubble("user", msg)

        state["messages"] = [HumanMessage(content=msg)]
        result = app.invoke(state, config)
        state.update(result)

        msgs = result.get("messages", [])
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                _print_bubble("agent", m.content)
                break

        _print_state_summary(result)

        if result.get("is_complete"):
            print("\n  [triage complete]")
            break


def main():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Add it to .env first.")
        return

    scenario_1 = [
        "Hi, my 6-year-old has had a fever since yesterday and threw up once at dinner. I'm worried.",
        "Temperature is 101.8 right now. He's awake and talking to me normally, breathing fine. "
        "He's drinking water like he usually does, just not eating as much solid food.",
        "He just peed about 3 hours ago, normal amount. He's on amoxicillin for an ear infection.",
    ]

    scenario_2 = [
        "My 6-year-old is really sick. High fever and she threw up once.",
        "Her temperature is 103.5. She's really out of it — barely responding when I call her. "
        "She won't drink anything. Her breathing seems okay though. "
        "There's a stomach bug going around her school.",
        "I don't think she's peed in about 8 hours.",
    ]

    run_scenario(
        "SCENARIO 1 — Moderate fever, alert, drinking → expect home_monitor",
        scenario_1,
        thread_id="live-s1",
    )
    run_scenario(
        "SCENARIO 2 — High fever, reduced alertness, not drinking → expect er_now",
        scenario_2,
        thread_id="live-s2",
    )


if __name__ == "__main__":
    main()
