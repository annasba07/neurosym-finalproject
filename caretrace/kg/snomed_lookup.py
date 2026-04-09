"""Placeholder SNOMED concept lookup.

Dict-based mapping for the skeleton. David replaces this with Neo4j
Cypher queries against AuraDB containing the real SNOMED slice.

The interface (normalize_concepts) stays the same regardless of backend.
"""

# Scoped SNOMED concepts relevant to pediatric febrile illness + GI + dehydration.
# Real SCTIDs from SNOMED CT.
SYMPTOM_TO_SNOMED = {
    "fever": {
        "sctid": "386661006",
        "term": "Fever",
        "ancestors": ["Finding of body temperature", "Clinical finding"],
    },
    "high_fever": {
        "sctid": "50177009",
        "term": "High fever",
        "ancestors": ["Fever", "Finding of body temperature", "Clinical finding"],
    },
    "vomiting": {
        "sctid": "422400008",
        "term": "Vomiting",
        "ancestors": ["Gastrointestinal finding", "Clinical finding"],
    },
    "repeated_vomiting": {
        "sctid": "424580008",
        "term": "Persistent vomiting",
        "ancestors": ["Vomiting", "Gastrointestinal finding"],
    },
    "decreased_urine_output": {
        "sctid": "267064002",
        "term": "Decreased urine output",
        "ancestors": ["Renal finding", "Clinical finding"],
    },
    "dehydration": {
        "sctid": "34095006",
        "term": "Dehydration",
        "ancestors": ["Fluid and electrolyte disorder", "Clinical finding"],
    },
    "lethargy": {
        "sctid": "214264003",
        "term": "Lethargy",
        "ancestors": ["Alteration of consciousness", "Neurological finding"],
    },
    "breathing_difficulty": {
        "sctid": "267036007",
        "term": "Dyspnea",
        "ancestors": ["Respiratory finding", "Clinical finding"],
    },
    "seizure": {
        "sctid": "91175000",
        "term": "Seizure",
        "ancestors": ["Neurological finding", "Clinical finding"],
    },
    "rash": {
        "sctid": "271807003",
        "term": "Eruption of skin",
        "ancestors": ["Skin finding", "Clinical finding"],
    },
    "acetaminophen": {
        "sctid": "387517004",
        "term": "Acetaminophen",
        "ancestors": ["Analgesic", "Antipyretic"],
    },
    "ibuprofen": {
        "sctid": "387207008",
        "term": "Ibuprofen",
        "ancestors": ["NSAID", "Anti-inflammatory agent"],
    },
    "amoxicillin": {
        "sctid": "27658006",
        "term": "Amoxicillin",
        "ancestors": ["Penicillin", "Antibiotic"],
    },
    "reduced_fluid_intake": {
        "sctid": "271795006",
        "term": "Reduced fluid intake",
        "ancestors": ["Finding of fluid intake", "Clinical finding"],
    },
}


def normalize_concepts(state: dict) -> list[dict]:
    """Map extracted clinical facts to SNOMED concepts.

    Args:
        state: ClinicalState dict with extracted clinical fields.

    Returns:
        List of matched SNOMED concept dicts with sctid, term, ancestors.
    """
    concepts = []

    # Fever
    if state.get("fever") == "yes":
        temp = state.get("temperature_f")
        if temp is not None and temp >= 104.0:
            concepts.append(SYMPTOM_TO_SNOMED["high_fever"])
        else:
            concepts.append(SYMPTOM_TO_SNOMED["fever"])

    # Vomiting
    vomiting = state.get("vomiting")
    if vomiting == "once":
        concepts.append(SYMPTOM_TO_SNOMED["vomiting"])
    elif vomiting == "repeated":
        concepts.append(SYMPTOM_TO_SNOMED["repeated_vomiting"])

    # Urine / dehydration signals
    urine_hours = state.get("urine_hours")
    if urine_hours is not None and urine_hours >= 8:
        concepts.append(SYMPTOM_TO_SNOMED["decreased_urine_output"])

    drinking = state.get("drinking")
    if drinking == "no":
        concepts.append(SYMPTOM_TO_SNOMED["reduced_fluid_intake"])
        concepts.append(SYMPTOM_TO_SNOMED["dehydration"])
    elif drinking == "some" and urine_hours is not None and urine_hours >= 8:
        concepts.append(SYMPTOM_TO_SNOMED["dehydration"])

    # Alertness
    if state.get("alert") == "no":
        concepts.append(SYMPTOM_TO_SNOMED["lethargy"])

    # Breathing
    if state.get("breathing_issue") == "yes":
        concepts.append(SYMPTOM_TO_SNOMED["breathing_difficulty"])

    # Seizure
    if state.get("seizure") == "yes":
        concepts.append(SYMPTOM_TO_SNOMED["seizure"])

    # Rash
    if state.get("rash") == "yes":
        concepts.append(SYMPTOM_TO_SNOMED["rash"])

    # Medications
    med = state.get("medications")
    if med and med.lower() not in ("none", "no"):
        med_lower = med.lower()
        if "amoxicillin" in med_lower:
            concepts.append(SYMPTOM_TO_SNOMED["amoxicillin"])
        elif "acetaminophen" in med_lower or "tylenol" in med_lower:
            concepts.append(SYMPTOM_TO_SNOMED["acetaminophen"])
        elif "ibuprofen" in med_lower or "motrin" in med_lower or "advil" in med_lower:
            concepts.append(SYMPTOM_TO_SNOMED["ibuprofen"])

    return concepts
