from knowledge_graph_agent.agent import KnowledgeRetrievalAgent

def kg_agent(state):
    facts = state["facts"]

    mentions = []
    if facts["fever"] == "yes":
        mentions.append("fever")
    if facts["alert"] == "reduced":
        mentions.append("lethargic")
    if facts["vomiting"] in ("once", "repeated"):
        mentions.append("vomiting")
    if facts["intake"] == "none":
        mentions.append("not drinking")
    if facts["urination"] == "none":
        mentions.append("not urinating")

    with KnowledgeRetrievalAgent() as agent:
        grounding = agent.ground_symptoms_batch(mentions)
        flags = agent.get_red_flags(grounding["all_sctids"], age_months=facts["age_months"])

    sctids = grounding["all_sctids"]

    state["facts"] = {
        "fever":      "yes" if "386661006" in sctids else facts["fever"],
        "alert":      "reduced" if "40917007" in sctids else facts["alert"],
        "vomiting":   facts["vomiting"],
        "intake":     "none" if "34095006" in sctids else facts["intake"],
        "urination":  "none" if "28442001" in sctids else facts["urination"],
        "age_months": facts["age_months"],
        "weight_kg":  facts["weight_kg"],
    }
    state["kg_concepts"] = sctids
    state["derived_flags"] = flags["red_flags"]

    return state
