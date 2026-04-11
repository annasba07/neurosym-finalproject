"""Knowledge agent — grounds symptoms and collects red flags.

Pipeline:
1. Ground raw_symptoms via KnowledgeAdapter (Neo4j or dict fallback)
2. If neo4j is live, also ask it for red flags keyed on the returned sctids
3. Always run dict-based fallback red flags against the raw state
   (covers infant fever, seizure, breathing, very high fever, rash+fever)
4. Merge both red-flag sources into state["kg_red_flags"]

This means the rules/safety layer always has a populated red-flag list
regardless of backend, and the demo notebook can show provenance per
red flag via the `source` field ('neo4j' | 'fallback').
"""

from caretrace.state import ClinicalState
from caretrace.kg.adapter import KnowledgeAdapter
from caretrace.kg.fallback_red_flags import check_fallback_red_flags


# Single module-level adapter so we don't re-probe Neo4j on every turn.
_adapter: KnowledgeAdapter | None = None


def _get_adapter() -> KnowledgeAdapter:
    global _adapter
    if _adapter is None:
        # CARETRACE_KG_BACKEND env var: "dict" | "neo4j" | "auto" (default)
        import os
        prefer = os.getenv("CARETRACE_KG_BACKEND", "auto")
        _adapter = KnowledgeAdapter(prefer=prefer)
    return _adapter


def set_adapter(adapter: KnowledgeAdapter) -> None:
    """Inject a specific adapter (used by tests to force dict mode)."""
    global _adapter
    _adapter = adapter


def normalize(state: ClinicalState) -> dict:
    """LangGraph node: ground symptoms + gather red flags."""
    adapter = _get_adapter()

    mentions = state.get("raw_symptoms") or []
    grounding = adapter.ground_symptoms(mentions) if mentions else {
        "backend":    adapter.backend,
        "results":    [],
        "all_sctids": [],
        "ungrounded": [],
    }

    # Neo4j-sourced red flags (empty list in dict mode)
    kg_flags = adapter.get_red_flags(grounding["all_sctids"])

    # Always run fallback red flags — they catch signals Alex's rules miss.
    # This is belt-and-suspenders safety: even in Neo4j mode, we cross-check.
    fb_flags = check_fallback_red_flags(state)

    # Merge, de-duplicate by rule_id (neo4j wins over fallback on collision)
    seen_ids = {f["rule_id"] for f in kg_flags}
    merged = list(kg_flags) + [f for f in fb_flags if f["rule_id"] not in seen_ids]

    return {
        "kg_backend":        grounding["backend"],
        "grounded_concepts": grounding["results"],
        "all_sctids":        grounding["all_sctids"],
        "kg_red_flags":      merged,
    }
