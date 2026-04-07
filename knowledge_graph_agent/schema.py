"""
schema.py
Defines all node labels, relationship types, and property keys used in the
CareTrace Neo4j knowledge graph. All other modules import from here so that
naming is consistent across the codebase.
"""

# ── Node Labels ────────────────────────────────────────────────────────────────

class NodeLabel:
    CONCEPT      = "Concept"       # SNOMED clinical concept (symptom / finding / disorder)
    MEDICATION   = "Medication"    # Drug with dosing metadata
    RED_FLAG     = "RedFlag"       # Hard escalation trigger
    LOCAL_CONTEXT = "LocalContext" # Optional epidemiological prior


# ── Relationship Types ─────────────────────────────────────────────────────────

class RelType:
    IS_A               = "IS_A"                # SNOMED hierarchy (child → parent)
    ASSOCIATED_WITH    = "ASSOCIATED_WITH"     # Concept → RedFlag
    TREATED_BY         = "TREATED_BY"          # Concept → Medication
    CONTRAINDICATED_IN = "CONTRAINDICATED_IN"  # Medication → Concept


# ── Property Keys ──────────────────────────────────────────────────────────────

class ConceptProps:
    SCTID       = "sctid"        # SNOMED concept ID  e.g. "386661006"
    FSN         = "fsn"          # Fully specified name e.g. "Fever (finding)"
    SYNONYMS    = "synonyms"     # List of common aliases e.g. ["fever", "pyrexia", "high temp"]
    TYPE        = "type"         # "symptom" | "finding" | "disorder"


class MedicationProps:
    NAME               = "name"                # e.g. "ibuprofen"
    FORMULATION        = "formulation"         # e.g. "oral suspension 100mg/5ml"
    DOSE_MG_PER_KG     = "dose_mg_per_kg"      # Standard dose
    MAX_DOSE_MG        = "max_dose_mg"         # Absolute ceiling per administration
    MAX_DAILY_DOSES    = "max_daily_doses"     # How many times per day
    MIN_INTERVAL_HOURS = "min_interval_hours"  # Minimum hours between doses
    MIN_AGE_MONTHS     = "min_age_months"      # Minimum age for use
    MIN_WEIGHT_KG      = "min_weight_kg"       # Minimum weight for use
    SOURCE             = "source"              # Provenance e.g. "AAP 2023"


class RedFlagProps:
    DESCRIPTION = "description"  # Human-readable trigger description
    DISPOSITION = "disposition"  # "ED_NOW" | "URGENT_CARE" | "HOME"
    RULE_ID     = "rule_id"      # Maps to PyDatalog rule e.g. "RF_001"


class LocalContextProps:
    SIGNAL  = "signal"   # e.g. "RSV_high_prevalence"
    WEIGHT  = "weight"   # Float 0–1, how strongly to weight this prior
    SOURCE  = "source"   # e.g. "CDC_surveillance"


# ── Disposition Values (used by RedFlag and Safety Logic Agent) ────────────────

class Disposition:
    ED_NOW       = "ED_NOW"       # Emergency department immediately
    URGENT_CARE  = "URGENT_CARE"  # Same-day urgent evaluation
    HOME         = "HOME"         # Home management with safety-netting


# ── Concept Types ──────────────────────────────────────────────────────────────

class ConceptType:
    SYMPTOM  = "symptom"
    FINDING  = "finding"
    DISORDER = "disorder"