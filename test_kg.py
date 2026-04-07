# test_kg.py
from knowledge_graph_agent.agent import KnowledgeRetrievalAgent

with KnowledgeRetrievalAgent() as agent:

    # ─────────────────────────────────────────────
    # SCENARIO 1 — Moderate fever, home management
    # 6yo ~20kg, on amoxicillin, vomited once,
    # urinated recently, responsive, sipping fluids
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("SCENARIO 1 — Moderate fever, home management")
    print("=" * 60)

    s1_symptoms = ["fever", "vomiting", "tired"]
    s1_grounding = agent.ground_symptoms_batch(s1_symptoms)
    print("Grounding:")
    for mention, result in s1_grounding["results"].items():
        print(f"  '{mention}' → grounded={result['grounded']}")
    if s1_grounding["ungrounded"]:
        print(f"  UNGROUNDED: {s1_grounding['ungrounded']}")
    print()

    s1_flags = agent.get_red_flags(s1_grounding["all_sctids"])
    print("Red flags:", s1_flags)
    # expect: no ED_NOW — RF_009 (vomiting) at URGENT_CARE at most
    print()

    s1_ibu_contra = agent.get_contraindications("ibuprofen", s1_grounding["all_sctids"])
    print("Ibuprofen contraindicated:", s1_ibu_contra)
    # expect: contraindicated=False — no dehydration signal
    print()

    s1_ibu_dose = agent.get_dosing("ibuprofen", 20.0)
    print("Ibuprofen dosing (20kg):", s1_ibu_dose)
    # expect: dose_mg=200, dose_ml=10.0
    print()

    s1_acet_dose = agent.get_dosing("acetaminophen", 20.0)
    print("Acetaminophen dosing (20kg):", s1_acet_dose)
    # expect: dose_mg=300, dose_ml=9.375
    print()

    # ─────────────────────────────────────────────
    # SCENARIO 2 — High fever, ER escalation
    # 6yo ~20kg, barely responding, not drinking,
    # no urination since afternoon, school virus context
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("SCENARIO 2 — High fever, ER escalation")
    print("=" * 60)

    s2_symptoms = ["fever", "vomiting", "lethargic", "not urinating"]
    s2_grounding = agent.ground_symptoms_batch(s2_symptoms)
    print("Grounding:")
    for mention, result in s2_grounding["results"].items():
        print(f"  '{mention}' → grounded={result['grounded']}")
    if s2_grounding["ungrounded"]:
        print(f"  UNGROUNDED: {s2_grounding['ungrounded']}")
    print()

    s2_flags = agent.get_red_flags(s2_grounding["all_sctids"])
    print("Red flags:", s2_flags)
    # expect: ED_NOW — RF_005 (lethargy) + RF_007 (dehydration via IS_A chain)
    print()

    s2_ibu_contra = agent.get_contraindications("ibuprofen", s2_grounding["all_sctids"])
    print("Ibuprofen contraindicated:", s2_ibu_contra)
    # expect: contraindicated=True — dehydration present via IS_A
    print()

    s2_safe_meds = agent.get_safe_medications(s2_grounding["all_sctids"], weight_kg=20.0, age_months=72)
    print("Safe medications:")
    for med in s2_safe_meds:
        print(f"  {med['medication']}: safe={med['safe']}")
    # expect: ibuprofen safe=False, acetaminophen safe=True
    print()

    # ─────────────────────────────────────────────
    # GROUNDING EDGE CASES
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("EDGE CASES")
    print("=" * 60)

    # "tired" / "wiped out" — known gap, expect ungrounded
    for term in ["tired", "wiped out", "fatigue"]:
        result = agent.ground_symptom(term)
        print(f"  '{term}' → grounded={result['grounded']}")
    print()

    # IS_A chain audit for dehydration symptoms
    for term in ["not urinating", "dry lips", "sunken eyes"]:
        print(agent.explain_grounding(term))