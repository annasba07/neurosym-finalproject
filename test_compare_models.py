from src.rules import rules_agent
from llm_openrouter import llm_triage


MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "mistralai/mistral-7b-instruct"
]


def run_case(name, raw_input, facts):
    print("\n==============================")
    print(name)
    print("==============================")

    print("\nUser Input:")
    print(raw_input)

    print("\n--- LLM Outputs ---")

    for model in MODELS:
        try:
            output = llm_triage(raw_input, model)
            print(f"\n[{model}]")
            print(output)
        except Exception as e:
            print(f"\n[{model}] ERROR:", e)

    print("\n--- CareTrace Output ---")

    state = {"facts": facts}
    result = rules_agent(state)

    print("Decision:", result["decision"])
    print("Rules triggered:", result["rules_triggered"])


# =========================
# SCENARIO 1
# =========================
scenario_1_input = """
My 6-year-old has a fever, threw up once, and looks really wiped out.
He’s tired but answers me. He’s sipping water, not much though.
He peed earlier this evening.
"""

scenario_1_facts = {
    "fever": "yes",
    "alert": "normal",
    "vomiting": "once",
    "intake": "reduced",
    "urination": "normal"
}


# =========================
# SCENARIO 2
# =========================
scenario_2_input = """
My 6-year-old has a fever, threw up, and looks really wiped out.
He’s barely responding, just lying there. He doesn’t want to drink.
I don’t think he’s peed since this afternoon.
"""

scenario_2_facts = {
    "fever": "yes",
    "alert": "reduced",
    "vomiting": "repeated",
    "intake": "none",
    "urination": "none"
}


if __name__ == "__main__":
    run_case("SCENARIO 1 (HOME)", scenario_1_input, scenario_1_facts)
    run_case("SCENARIO 2 (ER)", scenario_2_input, scenario_2_facts)