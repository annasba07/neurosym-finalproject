from src.rules import rules_agent

# From Turn 3 from Lecture 20 example
from src.rules import rules_agent

def test_home_case():
    state = {
        "facts": {
            "fever": "yes",
            "alert": "normal",
            "vomiting": "unknown",
            "intake": "reduced",
            "urination": "unknown"
        }
    }

    result = rules_agent(state)

    print("\n--- TEST CASE ---")
    print("State:", state["facts"])
    print("Decision:", result["decision"])
    print("Observation predicates:", result["observation_predicates"])
    print("Concern predicates:", result["concern_predicates"])
    print("Rules triggered:", result["rules_triggered"])


if __name__ == "__main__":
    test_home_case()