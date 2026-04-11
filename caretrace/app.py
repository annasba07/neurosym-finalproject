"""Interactive CLI for CareTrace triage agent.

Run with: python -m caretrace.app
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from caretrace.graph import create_app
from caretrace.state import initial_state


def main():
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    print("=" * 60)
    print("CareTrace — Pediatric Fever Triage Agent")
    print("Describe your child's symptoms. Type 'quit' to exit.")
    print("Type 'debug' after a response to see the rule trace.")
    print("=" * 60)
    print()

    app, config = create_app(thread_id="cli-session")
    state = initial_state()
    show_debug = False

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "debug":
            show_debug = True
            # Re-display last state info
            _print_debug(state)
            continue

        # Add user message to state
        state["messages"] = [HumanMessage(content=user_input)]

        # Run the graph
        try:
            result = app.invoke(state, config)
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        # Update local state with results
        state.update(result)

        # Display the agent's response
        messages = result.get("messages", [])
        if messages:
            # Get the last AI message
            for msg in reversed(messages):
                if hasattr(msg, "content") and (
                    not hasattr(msg, "type") or msg.type == "ai"
                ):
                    print(f"\nCareTrace: {msg.content}\n")
                    break

        # Show debug if requested
        if show_debug:
            _print_debug(result)
            show_debug = False

        # Check if triage is complete
        if result.get("is_complete"):
            print("-" * 60)
            print("Triage complete. Start a new session or type 'quit'.")
            print("-" * 60)
            # Reset for new session
            state = initial_state()
            app, config = create_app(thread_id="cli-session-2")


def _print_debug(state: dict):
    """Print debug info: disposition, rule trace, facts, KG provenance."""
    print("\n--- DEBUG ---")
    print(f"  Phase:       {state.get('phase', '?')}")
    print(f"  Disposition: {state.get('disposition') or 'undecided'}")
    print(f"  Turn:        {state.get('turn', 0)}")
    print(f"  KG backend:  {state.get('kg_backend', '?')}")

    facts = state.get("facts", {}) or {}
    if facts:
        print("  Facts:")
        for k, v in facts.items():
            print(f"    {k} = {v}")

    missing = state.get("missing_required", [])
    if missing:
        print(f"  Missing required facts: {', '.join(missing)}")

    obs = state.get("observation_predicates", [])
    if obs:
        print(f"  Observations: {obs}")

    concerns = state.get("concern_predicates", [])
    if concerns:
        print(f"  Concerns:     {concerns}")

    rules = state.get("rules_triggered", [])
    if rules:
        print("  Rule trace:")
        for r in rules:
            print(f"    - {r}")

    red_flags = state.get("kg_red_flags", [])
    if red_flags:
        print("  Red flags:")
        for f in red_flags:
            src = f.get("source", "kg")
            print(f"    [{src}] {f['rule_id']}: {f['description']} → {f['disposition']}")

    grounded = state.get("grounded_concepts", [])
    if grounded:
        terms = [c.get("mention") or (c.get("concepts") or [{}])[0].get("fsn", "?") for c in grounded]
        print(f"  Grounded:    {terms}")

    positives = state.get("key_positives", [])
    negatives = state.get("key_negatives", [])
    if positives:
        print(f"  Key positives: {'; '.join(positives)}")
    if negatives:
        print(f"  Key negatives: {'; '.join(negatives)}")

    print("--- END DEBUG ---\n")


if __name__ == "__main__":
    main()
