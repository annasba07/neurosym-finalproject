"""
loader.py
Populates the CareTrace Neo4j knowledge graph with:
  - SNOMED concepts (symptoms, findings, disorders) for pediatric febrile illness
  - Medications with full dosing metadata
  - Red flags with dispositions and rule IDs
  - All relationships between them

Usage:
    python -m knowledge_graph_agent.loader
"""

import os
from neo4j import GraphDatabase
from .schema import (
    NodeLabel, RelType,
    ConceptProps, MedicationProps, RedFlagProps, LocalContextProps,
    Disposition, ConceptType
)

from dotenv import load_dotenv
load_dotenv()

# ── Connection ─────────────────────────────────────────────────────────────────

def get_driver():
    uri      = os.environ["NEO4J_URI"]
    user     = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, password))


# ── Constraints & Indexes ──────────────────────────────────────────────────────

CONSTRAINTS = [
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:{NodeLabel.CONCEPT})    REQUIRE c.sctid IS UNIQUE",
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (m:{NodeLabel.MEDICATION})  REQUIRE m.name  IS UNIQUE",
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (r:{NodeLabel.RED_FLAG})    REQUIRE r.rule_id IS UNIQUE",
]

def create_constraints(tx):
    for stmt in CONSTRAINTS:
        tx.run(stmt)


# ── Clinical Data ──────────────────────────────────────────────────────────────

# SNOMED concepts scoped to pediatric febrile illness + GI + dehydration
CONCEPTS = [
    # ── Fever cluster ──
    {
        ConceptProps.SCTID:    "386661006",
        ConceptProps.FSN:      "Fever (finding)",
        ConceptProps.SYNONYMS: ["fever", "pyrexia", "high temperature", "burning up", "hot", "febrile"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
    {
        ConceptProps.SCTID:    "248425001",
        ConceptProps.FSN:      "Febrile convulsion (disorder)",
        ConceptProps.SYNONYMS: ["febrile seizure", "fever seizure", "shaking with fever"],
        ConceptProps.TYPE:     ConceptType.DISORDER,
    },
    {
        ConceptProps.SCTID:    "271807003",
        ConceptProps.FSN:      "Rash (finding)",
        ConceptProps.SYNONYMS: ["rash", "spots", "blotches", "skin rash"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
    {
        ConceptProps.SCTID:    "271757001",
        ConceptProps.FSN:      "Petechiae (finding)",
        ConceptProps.SYNONYMS: ["petechiae", "pin-point spots", "tiny red spots", "non-blanching rash"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },

    # ── GI cluster ──
    {
        ConceptProps.SCTID:    "422587007",
        ConceptProps.FSN:      "Nausea (finding)",
        ConceptProps.SYNONYMS: ["nausea", "feeling sick", "queasy"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
    {
        ConceptProps.SCTID:    "422400008",
        ConceptProps.FSN:      "Vomiting (disorder)",
        ConceptProps.SYNONYMS: ["vomiting", "throwing up", "being sick", "puking", "vomits"],
        ConceptProps.TYPE:     ConceptType.DISORDER,
    },
    {
        ConceptProps.SCTID:    "62315008",
        ConceptProps.FSN:      "Diarrhea (finding)",
        ConceptProps.SYNONYMS: ["diarrhea", "diarrhoea", "loose stools", "watery stool", "runny poop"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
    {
        ConceptProps.SCTID:    "21522001",
        ConceptProps.FSN:      "Abdominal pain (finding)",
        ConceptProps.SYNONYMS: ["stomach pain", "belly ache", "abdominal pain", "tummy hurts"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },

    # ── Dehydration cluster ──
    {
        ConceptProps.SCTID:    "34095006",
        ConceptProps.FSN:      "Dehydration (disorder)",
        ConceptProps.SYNONYMS: ["dehydration", "dehydrated", "not drinking", "dry"],
        ConceptProps.TYPE:     ConceptType.DISORDER,
    },
    {
        ConceptProps.SCTID:    "28442001",
        ConceptProps.FSN:      "Decreased urine output (finding)",
        ConceptProps.SYNONYMS: ["not urinating", "no wet diapers", "low urine", "hasn't peed"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
    {
        ConceptProps.SCTID:    "1290011000",
        ConceptProps.FSN:      "Dry mucous membranes (finding)",
        ConceptProps.SYNONYMS: ["dry mouth", "dry lips", "tacky mouth", "parched"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
    {
        ConceptProps.SCTID:    "271594007",
        ConceptProps.FSN:      "Sunken eyes (finding)",
        ConceptProps.SYNONYMS: ["sunken eyes", "eyes look hollow", "sunken fontanelle"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },

    # ── Neurological / severity ──
    {
        ConceptProps.SCTID:    "40917007",
        ConceptProps.FSN:      "Lethargy (finding)",
        ConceptProps.SYNONYMS: ["lethargic", "very sleepy", "won't wake up", "listless", "unresponsive"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
    {
        ConceptProps.SCTID:    "162397003",
        ConceptProps.FSN:      "Neck stiffness (finding)",
        ConceptProps.SYNONYMS: ["stiff neck", "neck stiffness", "can't bend neck"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
    {
        ConceptProps.SCTID:    "230145002",
        ConceptProps.FSN:      "Difficulty breathing (finding)",
        ConceptProps.SYNONYMS: ["trouble breathing", "hard to breathe", "respiratory distress", "breathing fast", "working hard to breathe"],
        ConceptProps.TYPE:     ConceptType.FINDING,
    },
]

# IS_A hierarchy (child_sctid → parent_sctid)
IS_A_EDGES = [
    ("248425001", "386661006"),   # Febrile convulsion IS_A Fever
    ("271757001", "271807003"),   # Petechiae IS_A Rash
    ("422587007", "422400008"),   # Nausea IS_A Vomiting (parent cluster)
    ("28442001",  "34095006"),    # Decreased urine output IS_A Dehydration
    ("1290011000","34095006"),    # Dry mucous membranes IS_A Dehydration
    ("271594007", "34095006"),    # Sunken eyes IS_A Dehydration
]

# Medications
MEDICATIONS = [
    {
        MedicationProps.NAME:               "acetaminophen",
        MedicationProps.FORMULATION:        "oral suspension 160mg/5ml",
        MedicationProps.DOSE_MG_PER_KG:     15,
        MedicationProps.MAX_DOSE_MG:        1000,
        MedicationProps.MAX_DAILY_DOSES:    5,
        MedicationProps.MIN_INTERVAL_HOURS: 4,
        MedicationProps.MIN_AGE_MONTHS:     2,
        MedicationProps.MIN_WEIGHT_KG:      3.0,
        MedicationProps.SOURCE:             "AAP 2023",
    },
    {
        MedicationProps.NAME:               "ibuprofen",
        MedicationProps.FORMULATION:        "oral suspension 100mg/5ml",
        MedicationProps.DOSE_MG_PER_KG:     10,
        MedicationProps.MAX_DOSE_MG:        600,
        MedicationProps.MAX_DAILY_DOSES:    4,
        MedicationProps.MIN_INTERVAL_HOURS: 6,
        MedicationProps.MIN_AGE_MONTHS:     6,
        MedicationProps.MIN_WEIGHT_KG:      5.0,
        MedicationProps.SOURCE:             "AAP 2023",
    },
]

# Red flags
RED_FLAGS = [
    {
        RedFlagProps.RULE_ID:     "RF_001",
        RedFlagProps.DESCRIPTION: "Fever in infant under 3 months",
        RedFlagProps.DISPOSITION: Disposition.ED_NOW,
    },
    {
        RedFlagProps.RULE_ID:     "RF_002",
        RedFlagProps.DESCRIPTION: "Febrile seizure",
        RedFlagProps.DISPOSITION: Disposition.ED_NOW,
    },
    {
        RedFlagProps.RULE_ID:     "RF_003",
        RedFlagProps.DESCRIPTION: "Non-blanching petechial rash with fever",
        RedFlagProps.DISPOSITION: Disposition.ED_NOW,
    },
    {
        RedFlagProps.RULE_ID:     "RF_004",
        RedFlagProps.DESCRIPTION: "Neck stiffness with fever",
        RedFlagProps.DISPOSITION: Disposition.ED_NOW,
    },
    {
        RedFlagProps.RULE_ID:     "RF_005",
        RedFlagProps.DESCRIPTION: "Lethargy or inability to rouse",
        RedFlagProps.DISPOSITION: Disposition.ED_NOW,
    },
    {
        RedFlagProps.RULE_ID:     "RF_006",
        RedFlagProps.DESCRIPTION: "Respiratory distress",
        RedFlagProps.DISPOSITION: Disposition.ED_NOW,
    },
    {
        RedFlagProps.RULE_ID:     "RF_007",
        RedFlagProps.DESCRIPTION: "Severe dehydration: no urine output > 8 hours, sunken eyes, dry mucous membranes",
        RedFlagProps.DISPOSITION: Disposition.ED_NOW,
    },
    {
        RedFlagProps.RULE_ID:     "RF_008",
        RedFlagProps.DESCRIPTION: "Fever persisting > 5 days",
        RedFlagProps.DISPOSITION: Disposition.URGENT_CARE,
    },
    {
        RedFlagProps.RULE_ID:     "RF_009",
        RedFlagProps.DESCRIPTION: "Vomiting preventing oral rehydration",
        RedFlagProps.DISPOSITION: Disposition.URGENT_CARE,
    },
    {
        RedFlagProps.RULE_ID:     "RF_010",
        RedFlagProps.DESCRIPTION: "Moderate dehydration: no urine > 6 hours",
        RedFlagProps.DISPOSITION: Disposition.URGENT_CARE,
    },
]

# Concept → RedFlag associations (sctid, rule_id)
CONCEPT_REDFLAG_EDGES = [
    ("248425001", "RF_002"),   # Febrile convulsion → RF_002
    ("271757001", "RF_003"),   # Petechiae → RF_003
    ("386661006", "RF_001"),   # Fever → RF_001 (age check done in rules)
    ("162397003", "RF_004"),   # Neck stiffness → RF_004
    ("40917007",  "RF_005"),   # Lethargy → RF_005
    ("230145002", "RF_006"),   # Difficulty breathing → RF_006
    ("34095006",  "RF_007"),   # Dehydration → RF_007
    ("422400008", "RF_009"),   # Vomiting → RF_009
    ("28442001",  "RF_010"),   # Decreased urine output → RF_010
]

# Concept → Medication (what is used to treat it)
TREATED_BY_EDGES = [
    ("386661006", "acetaminophen"),   # Fever treated by acetaminophen
    ("386661006", "ibuprofen"),       # Fever treated by ibuprofen
]

# Medication → Concept contraindications
CONTRAINDICATED_IN_EDGES = [
    ("ibuprofen", "34095006"),   # ibuprofen contraindicated in dehydration
]


# ── Load Functions ─────────────────────────────────────────────────────────────

def load_concepts(tx):
    for c in CONCEPTS:
        tx.run(
            f"""
            MERGE (n:{NodeLabel.CONCEPT} {{sctid: $sctid}})
            SET n.fsn      = $fsn,
                n.synonyms = $synonyms,
                n.type     = $type
            """,
            sctid=c[ConceptProps.SCTID],
            fsn=c[ConceptProps.FSN],
            synonyms=c[ConceptProps.SYNONYMS],
            type=c[ConceptProps.TYPE],
        )

def load_is_a_edges(tx):
    for child_sctid, parent_sctid in IS_A_EDGES:
        tx.run(
            f"""
            MATCH (child:{NodeLabel.CONCEPT}  {{sctid: $child_sctid}})
            MATCH (parent:{NodeLabel.CONCEPT} {{sctid: $parent_sctid}})
            MERGE (child)-[:{RelType.IS_A}]->(parent)
            """,
            child_sctid=child_sctid,
            parent_sctid=parent_sctid,
        )

def load_medications(tx):
    for m in MEDICATIONS:
        tx.run(
            f"""
            MERGE (n:{NodeLabel.MEDICATION} {{name: $name}})
            SET n.formulation        = $formulation,
                n.dose_mg_per_kg     = $dose_mg_per_kg,
                n.max_dose_mg        = $max_dose_mg,
                n.max_daily_doses    = $max_daily_doses,
                n.min_interval_hours = $min_interval_hours,
                n.min_age_months     = $min_age_months,
                n.min_weight_kg      = $min_weight_kg,
                n.source             = $source
            """,
            **{k.replace(".", "_"): v for k, v in m.items()},
        )

def load_red_flags(tx):
    for r in RED_FLAGS:
        tx.run(
            f"""
            MERGE (n:{NodeLabel.RED_FLAG} {{rule_id: $rule_id}})
            SET n.description = $description,
                n.disposition = $disposition
            """,
            rule_id=r[RedFlagProps.RULE_ID],
            description=r[RedFlagProps.DESCRIPTION],
            disposition=r[RedFlagProps.DISPOSITION],
        )

def load_concept_redflag_edges(tx):
    for sctid, rule_id in CONCEPT_REDFLAG_EDGES:
        tx.run(
            f"""
            MATCH (c:{NodeLabel.CONCEPT}  {{sctid: $sctid}})
            MATCH (r:{NodeLabel.RED_FLAG} {{rule_id: $rule_id}})
            MERGE (c)-[:{RelType.ASSOCIATED_WITH}]->(r)
            """,
            sctid=sctid,
            rule_id=rule_id,
        )

def load_treated_by_edges(tx):
    for sctid, med_name in TREATED_BY_EDGES:
        tx.run(
            f"""
            MATCH (c:{NodeLabel.CONCEPT}    {{sctid: $sctid}})
            MATCH (m:{NodeLabel.MEDICATION} {{name: $med_name}})
            MERGE (c)-[:{RelType.TREATED_BY}]->(m)
            """,
            sctid=sctid,
            med_name=med_name,
        )

def load_contraindicated_edges(tx):
    for med_name, sctid in CONTRAINDICATED_IN_EDGES:
        tx.run(
            f"""
            MATCH (m:{NodeLabel.MEDICATION} {{name: $med_name}})
            MATCH (c:{NodeLabel.CONCEPT}    {{sctid: $sctid}})
            MERGE (m)-[:{RelType.CONTRAINDICATED_IN}]->(c)
            """,
            med_name=med_name,
            sctid=sctid,
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def load_all():
    driver = get_driver()
    with driver.session() as session:
        print("Creating constraints...")
        session.execute_write(create_constraints)

        print("Loading concepts...")
        session.execute_write(load_concepts)

        print("Loading IS_A edges...")
        session.execute_write(load_is_a_edges)

        print("Loading medications...")
        session.execute_write(load_medications)

        print("Loading red flags...")
        session.execute_write(load_red_flags)

        print("Loading concept → red flag edges...")
        session.execute_write(load_concept_redflag_edges)

        print("Loading treated_by edges...")
        session.execute_write(load_treated_by_edges)

        print("Loading contraindicated_in edges...")
        session.execute_write(load_contraindicated_edges)

    driver.close()
    print("✓ Knowledge graph loaded successfully.")


if __name__ == "__main__":
    load_all()