"""Fast checks for the notebook 06 integration and RAG output contract."""

import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
import rag  # noqa: E402


class RagChecks(unittest.TestCase):
    def test_rejects_ambiguous_judge_reply(self):
        with patch.object(rag.llm, "generate", return_value="NOT SUPPORTED"):
            self.assertFalse(rag.is_supported("claim", "source"))

    def test_citation_coverage_requires_each_sentence(self):
        check = rag.validate_citations(
            "[1] This first claim has evidence. The second claim lacks citations. "
            "[7] The third claim has a bad reference.", 5)
        self.assertEqual(check["cited_claims"], 1)
        self.assertEqual(check["claims"], 3)
        self.assertEqual(check["invalid_references"], [7])
        short = rag.validate_citations("Yes. [1] A longer claim is supported.", 5)
        self.assertEqual((short["cited_claims"], short["claims"]), (1, 2))

    def test_extractively_copied_sentence_is_verified_against_its_source(self):
        hits = pd.DataFrame([{"abstract": "This is a finding in the source abstract."}])
        synthesis = rag.evidence_synthesis(hits, n=1)
        self.assertEqual(rag.verify_evidence_synthesis(synthesis, hits), (1, 1))
        self.assertEqual(rag.verify_evidence_synthesis("[1] An invented finding is true.", hits), (0, 1))

    def test_context_and_citation_check_use_the_numbered_source(self):
        hits = pd.DataFrame([
            {"title": "First", "abstract": "The first paper studies retrieval. " + "x" * 1300},
            {"title": "Second", "abstract": "The second paper studies safety."},
        ])
        self.assertEqual(rag.hits_source(hits), rag.build_rag_prompt("question", hits).split(
            "Retrieved papers:\n\n", 1)[1].split("Using only these papers", 1)[0])
        self.assertNotIn("x" * 1300, rag.hits_source(hits))
        self.assertEqual(rag.exact_citation_support("[1] The first paper studies retrieval.", hits), (1, 1))
        self.assertEqual(rag.exact_citation_support("[2] The first paper studies retrieval.", hits), (0, 1))
        tail_hits = pd.DataFrame([{"title": "Long", "abstract": "Opening. " + "x" * 1300 +
                                  " The tail finding is reported."}])
        claim = "[1] The tail finding is reported."
        self.assertEqual(rag.exact_citation_support(claim, tail_hits), (1, 1))
        self.assertEqual(rag.exact_citation_support(claim, tail_hits, prompt_only=True), (0, 1))

    def test_wrong_number_of_papers_is_rejected_before_the_llm_judge(self):
        with patch.object(rag, "is_supported", return_value=True) as judge:
            self.assertEqual(rag.groundedness("All four papers study safety.", "source", expected_papers=5), (0, 1))
            judge.assert_not_called()


class DemoChecks(unittest.TestCase):
    def test_demo_uses_selected_components_and_flags_fallback(self):
        notebook = json.loads((ROOT / "codes" / "06_end_to_end_demo.ipynb").read_text())
        source = "".join(notebook["cells"][6]["source"])
        hits = pd.DataFrame([{
            "id": "p2", "title": "Related paper", "abstract": "A sourced finding is reported.",
            "primary_category": "cs.AI", "score": 0.8,
        }])
        calls = []

        extraction = SimpleNamespace(
            extract=lambda text: {"key_phrases": ["research topic"], "acronyms": [], "datasets": []},
            clean_text=lambda text: text.strip(),
            llm_extract=lambda text, shots=3: calls.append("llm_extract") or
            {"methods": ["MethodX"], "datasets": [], "tasks": []},
            hybrid_artefacts=lambda names, text: calls.append("hybrid") or names,
        )
        llm = SimpleNamespace(
            is_cached=lambda *args, **kwargs: True,
            generate=lambda *args, **kwargs: "generated answer",
        )
        rag_stub = SimpleNamespace(
            SYSTEM="system", GUARD="guard",
            build_rag_prompt=lambda query, rows: "prompt",
            parse_sections=lambda reply: {"synthesis": "Unsupported claim.", "gap": "Unknown.",
                                          "recommendations": "[1] Related paper", "format_ok": True,
                                          "gap_from_fallback": False},
            groundedness=lambda text, source, expected_papers=None: (1, 1) if text.startswith("[1]") else (0, 1),
            hits_source=lambda rows: "source",
            validate_citations=lambda text, n: {"cited_claims": int(text.startswith("[1]")),
                                                "claims": 1, "coverage": float(text.startswith("[1]")),
                                                "invalid_references": []},
            exact_citation_support=lambda text, rows, prompt_only=False: (1, 1) if text.startswith("[1]") else (0, 1),
            evidence_synthesis=lambda rows: calls.append("fallback") or "[1] A sourced finding is reported.",
            verify_evidence_synthesis=lambda text, rows: (1, 1),
            summarise_abstract=lambda text: calls.append("summary") or "A short paper summary.",
        )
        namespace = {
            "time": time, "ex": extraction, "clf": SimpleNamespace(predict=lambda texts: ["cs.AI"]),
            "HAVE_CLASSIFIER": True, "retrieval": SimpleNamespace(recommend=lambda pid, k: hits),
            "retrieve": lambda query, k: hits, "rag": rag_stub, "llm": llm,
        }
        exec(source, namespace)
        result = namespace["analyse"]("Input abstract describing a method.", paper_id="p1")

        self.assertIn("llm_extract", calls)
        self.assertIn("hybrid", calls)
        self.assertIn("summary", calls)
        self.assertIn("fallback", calls)
        self.assertEqual(result["entities"], ["MethodX"])
        self.assertEqual(result["summary"], "A short paper summary.")
        self.assertTrue(result["synthesis_fallback"])
        self.assertEqual(result["citation_coverage"], 1.0)
        self.assertEqual(result["grounded_share"], 1.0)
        self.assertEqual(result["synthesis_verification"], "exact source match")
        self.assertFalse(result["gap_verified"])
        self.assertFalse(result["recommendations_verified"])

        calls.clear()
        short_query = namespace["analyse"]("graph neural networks")
        self.assertNotIn("llm_extract", calls)
        self.assertNotIn("summary", calls)
        self.assertEqual(short_query["extraction_method"], "regex for short topic query")
        self.assertIsNone(short_query["summary"])

        pasted_abstract = "This paper describes a reproducible method and its evaluation. " * 8
        self_hits = pd.concat([pd.DataFrame([{
            "id": "p1", "title": "Input paper", "abstract": pasted_abstract,
            "primary_category": "cs.AI", "score": 1.0,
        }]), hits], ignore_index=True)
        namespace["retrieve"] = lambda query, k: self_hits
        pasted = namespace["analyse"](pasted_abstract)
        self.assertTrue(pasted["self_match_excluded"])
        self.assertEqual([paper["id"] for paper in pasted["papers"]], ["p2"])


if __name__ == "__main__":
    unittest.main()
