"""
queries.py
All Cypher queries used by the CareTrace Knowledge Retrieval Agent.
Each function takes a Neo4j session and returns clean Python dicts.
"""

from .schema import NodeLabel, RelType


# ── Symptom Grounding ──────────────────────────────────────────────────────────

def ground_symptom(session, text_mention: str) -> list[dict]:
    """
    Match a free-text symptom mention to SNOMED concepts via synonym list.
    Returns the matched concept(s) plus all ancestors via IS_A.

    Example:
        ground_symptom(session, "burning up")
        → [{"sctid": "386661006", "fsn": "Fever (finding)", "type": "finding", "depth": 0}]
    """
    result = session.run(
        f"""
        // Match concept by synonym (case-insensitive)
        MATCH (c:{NodeLabel.CONCEPT})
        WHERE any(s IN c.synonyms WHERE toLower(s) = toLower($mention))

        // Walk IS_A chain to collect ancestors
        OPTIONAL MATCH path = (c)-[:{RelType.IS_A}*1..5]->(ancestor:{NodeLabel.CONCEPT})

        WITH c,
             collect(DISTINCT {{
                 sctid: ancestor.sctid,
                 fsn:   ancestor.fsn,
                 type:  ancestor.type,
                 depth: length(path)
             }}) AS ancestors

        RETURN
            c.sctid    AS sctid,
            c.fsn      AS fsn,
            c.type     AS type,
            0          AS depth,
            ancestors
        """,
        mention=text_mention,
    )

    rows = []
    for record in result:
        rows.append({
            "sctid":     record["sctid"],
            "fsn":       record["fsn"],
            "type":      record["type"],
            "depth":     record["depth"],
            "ancestors": record["ancestors"],
        })
    return rows


# ── Red Flag Retrieval ─────────────────────────────────────────────────────────

def get_red_flags(session, sctids: list[str]) -> list[dict]:
    """
    Given a list of grounded SNOMED concept IDs (including ancestors),
    return any red flags associated with them.

    Example:
        get_red_flags(session, ["271757001", "271807003"])
        → [{"rule_id": "RF_003", "description": "Non-blanching petechial rash...", "disposition": "ED_NOW"}]
    """
    result = session.run(
        f"""
        MATCH (c:{NodeLabel.CONCEPT})-[:{RelType.ASSOCIATED_WITH}]->(r:{NodeLabel.RED_FLAG})
        WHERE c.sctid IN $sctids
        RETURN DISTINCT
            r.rule_id     AS rule_id,
            r.description AS description,
            r.disposition AS disposition,
            c.fsn         AS triggered_by
        ORDER BY r.disposition  // ED_NOW sorts before URGENT_CARE alphabetically
        """,
        sctids=sctids,
    )

    return [
        {
            "rule_id":      record["rule_id"],
            "description":  record["description"],
            "disposition":  record["disposition"],
            "triggered_by": record["triggered_by"],
        }
        for record in result
    ]


# ── Dosing Retrieval ───────────────────────────────────────────────────────────

def get_dosing(session, medication_name: str, weight_kg: float) -> dict | None:
    """
    Return a computed dose for a given medication and child weight.
    Enforces min weight, min age check data, and absolute dose ceiling.
    Returns None if medication not found or weight below minimum.

    Example:
        get_dosing(session, "ibuprofen", 14.0)
        → {
            "medication": "ibuprofen",
            "formulation": "oral suspension 100mg/5ml",
            "dose_mg": 140.0,
            "dose_ml": 7.0,
            "max_dose_mg": 600,
            "min_interval_hours": 6,
            "max_daily_doses": 4,
            "min_age_months": 6,
            "source": "AAP 2023",
            "safe_for_weight": True
          }
    """
    result = session.run(
        f"""
        MATCH (m:{NodeLabel.MEDICATION} {{name: $name}})
        RETURN
            m.name               AS name,
            m.formulation        AS formulation,
            m.dose_mg_per_kg     AS dose_mg_per_kg,
            m.max_dose_mg        AS max_dose_mg,
            m.max_daily_doses    AS max_daily_doses,
            m.min_interval_hours AS min_interval_hours,
            m.min_age_months     AS min_age_months,
            m.min_weight_kg      AS min_weight_kg,
            m.source             AS source
        """,
        name=medication_name.lower(),
    )

    record = result.single()
    if not record:
        return None

    safe_for_weight = weight_kg >= record["min_weight_kg"]

    # Calculate dose — cap at absolute maximum
    raw_dose_mg  = record["dose_mg_per_kg"] * weight_kg
    dose_mg      = min(raw_dose_mg, record["max_dose_mg"])

    # Calculate volume from formulation string e.g. "oral suspension 100mg/5ml"
    dose_ml = _compute_dose_ml(record["formulation"], dose_mg)

    return {
        "medication":        record["name"],
        "formulation":       record["formulation"],
        "dose_mg":           round(dose_mg, 1),
        "dose_ml":           round(dose_ml, 1) if dose_ml else None,
        "max_dose_mg":       record["max_dose_mg"],
        "min_interval_hours":record["min_interval_hours"],
        "max_daily_doses":   record["max_daily_doses"],
        "min_age_months":    record["min_age_months"],
        "source":            record["source"],
        "safe_for_weight":   safe_for_weight,
    }


def _compute_dose_ml(formulation: str, dose_mg: float) -> float | None:
    """
    Parse concentration from formulation string and convert mg → ml.
    Handles strings like '160mg/5ml' or '100mg/5ml'.
    """
    import re
    match = re.search(r"(\d+(?:\.\d+)?)mg/(\d+(?:\.\d+)?)ml", formulation)
    if not match:
        return None
    mg_per_unit = float(match.group(1))
    ml_per_unit = float(match.group(2))
    concentration = mg_per_unit / ml_per_unit  # mg per ml
    return dose_mg / concentration


# ── Contraindication Check ─────────────────────────────────────────────────────

def get_contraindications(session, medication_name: str, sctids: list[str]) -> list[dict]:
    """
    Check whether a medication is contraindicated given the patient's
    grounded concept IDs (symptoms + ancestors).

    Example:
        get_contraindications(session, "ibuprofen", ["34095006"])
        → [{"medication": "ibuprofen", "contraindicated_in": "Dehydration (disorder)", "sctid": "34095006"}]
    """
    result = session.run(
        f"""
        MATCH (m:{NodeLabel.MEDICATION} {{name: $name}})
              -[:{RelType.CONTRAINDICATED_IN}]->
              (c:{NodeLabel.CONCEPT})
        WHERE c.sctid IN $sctids
        RETURN
            m.name  AS medication,
            c.fsn   AS contraindicated_in,
            c.sctid AS sctid
        """,
        name=medication_name.lower(),
        sctids=sctids,
    )

    return [
        {
            "medication":        record["medication"],
            "contraindicated_in":record["contraindicated_in"],
            "sctid":             record["sctid"],
        }
        for record in result
    ]


# ── Concept Lookup (utility) ───────────────────────────────────────────────────

def get_concept_by_sctid(session, sctid: str) -> dict | None:
    """Fetch a single concept by SNOMED ID. Used for audit trail."""
    result = session.run(
        f"""
        MATCH (c:{NodeLabel.CONCEPT} {{sctid: $sctid}})
        RETURN c.sctid AS sctid, c.fsn AS fsn, c.type AS type, c.synonyms AS synonyms
        """,
        sctid=sctid,
    )
    record = result.single()
    if not record:
        return None
    return {
        "sctid":    record["sctid"],
        "fsn":      record["fsn"],
        "type":     record["type"],
        "synonyms": record["synonyms"],
    }


def get_all_ancestors(session, sctid: str) -> list[dict]:
    """
    Return all ancestors of a concept via IS_A chain.
    Used to expand a grounded concept before red flag / contraindication checks.
    """
    result = session.run(
        f"""
        MATCH (c:{NodeLabel.CONCEPT} {{sctid: $sctid}})
        OPTIONAL MATCH path = (c)-[:{RelType.IS_A}*1..5]->(ancestor:{NodeLabel.CONCEPT})
        RETURN DISTINCT
            ancestor.sctid AS sctid,
            ancestor.fsn   AS fsn,
            length(path)   AS depth
        ORDER BY depth
        """,
        sctid=sctid,
    )
    return [
        {"sctid": r["sctid"], "fsn": r["fsn"], "depth": r["depth"]}
        for r in result
        if r["sctid"] is not None
    ]