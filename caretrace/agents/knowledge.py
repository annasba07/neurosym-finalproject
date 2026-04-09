"""Knowledge agent — normalizes extracted facts to SNOMED concepts via KG.

Thin wrapper around kg/snomed_lookup.py. When David's Neo4j backend
is ready, swap the import — the interface stays the same.
"""

from caretrace.state import ClinicalState
from caretrace.kg.snomed_lookup import normalize_concepts


def normalize(state: ClinicalState) -> dict:
    """LangGraph node: map clinical facts to SNOMED concepts.

    Reads extracted clinical fields from state, calls the KG lookup,
    and writes normalized concepts back to state.
    """
    concepts = normalize_concepts(state)
    return {"snomed_concepts": concepts}
