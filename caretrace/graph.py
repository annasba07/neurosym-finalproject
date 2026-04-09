"""LangGraph StateGraph wiring for the CareTrace triage pipeline.

Flow:
    START → interpret → normalize → evaluate_rules → route
    route:
      disposition is set → explain → END
      missing fields     → ask_followup → END (wait for next user turn)
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from caretrace.state import ClinicalState, initial_state
from caretrace.agents.interpretation import interpret
from caretrace.agents.knowledge import normalize
from caretrace.agents.safety import evaluate_rules
from caretrace.agents.explanation import explain, ask_followup


def _route_after_rules(state: ClinicalState) -> str:
    """Conditional edge: decide whether to explain or ask follow-up."""
    disposition = state.get("disposition")
    if disposition is not None:
        return "explain"
    return "ask_followup"


def build_graph() -> StateGraph:
    """Construct the CareTrace triage graph."""
    graph = StateGraph(ClinicalState)

    # Add nodes
    graph.add_node("interpret", interpret)
    graph.add_node("normalize", normalize)
    graph.add_node("evaluate_rules", evaluate_rules)
    graph.add_node("explain", explain)
    graph.add_node("ask_followup", ask_followup)

    # Linear edges: interpret → normalize → evaluate_rules
    graph.set_entry_point("interpret")
    graph.add_edge("interpret", "normalize")
    graph.add_edge("normalize", "evaluate_rules")

    # Conditional edge after rules
    graph.add_conditional_edges(
        "evaluate_rules",
        _route_after_rules,
        {"explain": "explain", "ask_followup": "ask_followup"},
    )

    # Terminal edges
    graph.add_edge("explain", END)
    graph.add_edge("ask_followup", END)

    return graph


def create_app(thread_id: str = "default"):
    """Create a compiled graph with memory for multi-turn conversation.

    Args:
        thread_id: Unique ID for this triage session.

    Returns:
        Tuple of (compiled_graph, config) ready for .invoke().
    """
    graph = build_graph()
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": thread_id}}
    return app, config
