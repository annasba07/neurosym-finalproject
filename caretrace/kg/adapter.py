"""Knowledge graph adapter with Neo4j + dict fallback.

Primary backend is David's knowledge_graph_agent.KnowledgeRetrievalAgent
(live Neo4j). If the import fails, env vars are missing, or the
connection raises, we fall back to the dict-based snomed_lookup so the
full pipeline still runs end-to-end.

Every call returns a dict with a `backend` field so the demo notebook
can surface which backend served the request.
"""

from __future__ import annotations

import os
from typing import Optional

from caretrace.kg.snomed_lookup import SYMPTOM_TO_SNOMED


# Try to pull David's live agent, but don't fail import if unavailable.
try:
    from knowledge_graph_agent.agent import KnowledgeRetrievalAgent  # type: ignore
    _NEO4J_IMPORTABLE = True
except Exception:  # noqa: BLE001 — we truly want to swallow any import failure
    KnowledgeRetrievalAgent = None  # type: ignore
    _NEO4J_IMPORTABLE = False


def _neo4j_env_present() -> bool:
    return all(os.getenv(k) for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"))


# Map free-text mention → dict-backend symptom key.
_MENTION_TO_DICT_KEY = {
    "fever": "fever",
    "high fever": "high_fever",
    "vomiting": "vomiting",
    "vomited once": "vomiting",
    "repeated vomiting": "repeated_vomiting",
    "persistent vomiting": "repeated_vomiting",
    "lethargy": "lethargy",
    "lethargic": "lethargy",
    "reduced responsiveness": "lethargy",
    "decreased urine output": "decreased_urine_output",
    "low urine output": "decreased_urine_output",
    "reduced fluid intake": "reduced_fluid_intake",
    "poor fluid intake": "reduced_fluid_intake",
    "dehydration": "dehydration",
    "breathing difficulty": "breathing_difficulty",
    "trouble breathing": "breathing_difficulty",
    "seizure": "seizure",
    "rash": "rash",
    "acetaminophen": "acetaminophen",
    "tylenol": "acetaminophen",
    "ibuprofen": "ibuprofen",
    "motrin": "ibuprofen",
    "amoxicillin": "amoxicillin",
}


class KnowledgeAdapter:
    """Uniform KG interface. Uses Neo4j when possible, dict otherwise."""

    def __init__(self, prefer: str = "auto") -> None:
        """
        Args:
            prefer: "auto" (try neo4j, fall back to dict), "neo4j" (fail if
                unavailable), or "dict" (always use dict backend).
        """
        self._neo4j_agent: Optional[object] = None
        self.backend: str = "dict"

        if prefer == "dict":
            return

        if prefer in ("auto", "neo4j") and _NEO4J_IMPORTABLE and _neo4j_env_present():
            try:
                self._neo4j_agent = KnowledgeRetrievalAgent()  # type: ignore[call-arg]
                # Best-effort smoke check: run a trivial ground to ensure session works.
                _ = self._neo4j_agent.ground_symptom("fever")  # type: ignore[union-attr]
                self.backend = "neo4j"
            except Exception as exc:  # noqa: BLE001
                if prefer == "neo4j":
                    raise RuntimeError(f"Neo4j required but unavailable: {exc}") from exc
                self._neo4j_agent = None
                self.backend = "dict"
        elif prefer == "neo4j":
            raise RuntimeError(
                "Neo4j required but knowledge_graph_agent import or env vars unavailable."
            )

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "KnowledgeAdapter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._neo4j_agent is not None:
            try:
                self._neo4j_agent.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self._neo4j_agent = None

    # ── Grounding ─────────────────────────────────────────────────────────────

    def ground_symptoms(self, mentions: list[str]) -> dict:
        """Ground a list of free-text mentions to SNOMED concepts.

        Returns:
            {
                "backend":  "neo4j" | "dict",
                "results":  [{mention, concepts, all_sctids, grounded}, ...],
                "all_sctids": [deduped flat list],
                "ungrounded": [mentions with no match],
            }
        """
        if self._neo4j_agent is not None:
            try:
                batch = self._neo4j_agent.ground_symptoms_batch(mentions)  # type: ignore[union-attr]
                return {
                    "backend":    "neo4j",
                    "results":    [
                        {**batch["results"][m], "mention": m}
                        for m in mentions
                        if m in batch["results"]
                    ],
                    "all_sctids": batch["all_sctids"],
                    "ungrounded": batch["ungrounded"],
                }
            except Exception:  # noqa: BLE001
                # Live graph hiccup mid-session — fall back silently.
                self.backend = "dict"

        # Dict fallback
        results: list[dict] = []
        ungrounded: list[str] = []
        all_sctids: set[str] = set()

        for mention in mentions:
            key = _MENTION_TO_DICT_KEY.get(mention.lower().strip())
            if key is None or key not in SYMPTOM_TO_SNOMED:
                ungrounded.append(mention)
                continue
            entry = SYMPTOM_TO_SNOMED[key]
            concept = {
                "sctid":     entry["sctid"],
                "fsn":       entry["term"],
                "type":      "finding",
                "depth":     0,
                "ancestors": [{"fsn": a, "sctid": None} for a in entry["ancestors"]],
            }
            results.append({
                "mention":    mention,
                "concepts":   [concept],
                "all_sctids": [entry["sctid"]],
                "grounded":   True,
            })
            all_sctids.add(entry["sctid"])

        return {
            "backend":    "dict",
            "results":    results,
            "all_sctids": list(all_sctids),
            "ungrounded": ungrounded,
        }

    # ── Red flags ─────────────────────────────────────────────────────────────

    def get_red_flags(self, sctids: list[str]) -> list[dict]:
        """Fetch red flags for a list of sctids. Only meaningful in neo4j mode.

        Each returned dict has a `source` field ('neo4j' or 'fallback') and
        disposition in Alex's vocabulary (er_now / urgent_eval / home_monitor).
        """
        if self._neo4j_agent is None or not sctids:
            return []

        try:
            result = self._neo4j_agent.get_red_flags(sctids)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return []

        normalized: list[dict] = []
        for r in result.get("red_flags", []):
            normalized.append({
                "rule_id":      r["rule_id"],
                "description":  r["description"],
                "disposition":  _david_to_alex(r["disposition"]),
                "triggered_by": r.get("triggered_by", ""),
                "source":       "neo4j",
            })
        return normalized


# ── Disposition vocabulary mapping ────────────────────────────────────────────

_DAVID_TO_ALEX = {
    "ED_NOW":      "er_now",
    "URGENT_CARE": "urgent_eval",
    "HOME":        "home_monitor",
}


def _david_to_alex(disposition: str) -> str:
    """Map David's KG dispositions (ED_NOW/URGENT_CARE/HOME) to Alex's vocabulary."""
    return _DAVID_TO_ALEX.get(disposition, disposition)
