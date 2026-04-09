"""Scripted end-to-end run of both clinical scenarios through the live LangGraph.

Uses the real Groq LLM for interpretation + explanation, pyDatalog for rules,
and the placeholder KG for concept lookup. Simulates a multi-turn conversation.

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
    missing = state.get("missing_required", [])
    if missing:
        print(f"    missing     : {missing}")
    rules = state.get("rules_fired", [])
    if rules:
        print("    rules fired :")
        for r in rules:
            print(f"      [{r['category']}] {r['reason']}")
    concepts = state.get("snomed_concepts", [])
    if concepts:
        print(f"    snomed      : {[c['term'] for c in concepts]}")
    print("  ----------------------")


def run_scenario(title: str, messages: list[str], thread_id: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    app, config = create_app(thread_id=thread_id)
    state = initial_state()

    for i, msg in enumerate(messages, 1):
        _print_bubble("user", msg)

        state["messages"] = [HumanMessage(content=msg)]
        result = app.invoke(state, config)
        state.update(result)

        # Find and print the latest AI message
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

    # Scenario 1: straightforward home management case (from Scenarios file)
    scenario_1 = [
        "Hi, my 6-year-old has had a fever since yesterday and threw up once at dinner. I'm worried.",
        "Temperature is 101.8 right now. He's awake and talking to me, breathing fine. "
        "He's sipping water but not eating much.",
        "He just took a little pee about 3 hours ago. He's on amoxicillin for an ear infection.",
    ]

    # Scenario 2: ER referral case
    scenario_2 = [
        "My 6-year-old is really sick. High fever and she threw up once.",
        "Her temperature is 103.5. She's really out of it — barely responding when I call her. "
        "She won't drink anything. Her breathing seems okay though. "
        "There's a stomach bug going around her school.",
        "I don't think she's peed in about 8 hours.",
    ]

    run_scenario(
        "SCENARIO 1 — Moderate fever, alert, drinking → expect HOME",
        scenario_1,
        thread_id="live-s1",
    )
    run_scenario(
        "SCENARIO 2 — High fever, not alert, not drinking → expect ER",
        scenario_2,
        thread_id="live-s2",
    )


if __name__ == "__main__":
    main()
