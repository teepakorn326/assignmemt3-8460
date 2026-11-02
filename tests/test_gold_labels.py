"""Data-integrity checks for NB02's provisional extraction labels."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from utils import load_sample  # noqa: E402


class GoldLabelChecks(unittest.TestCase):
    def test_labels_match_held_out_abstracts(self):
        labels = json.loads((ROOT / "codes/artifacts/02_gold_labels.json").read_text())
        split = json.loads((ROOT / "codes/data/split_ids.json").read_text())
        ids = [item["id"] for item in labels]
        self.assertEqual(len(ids), 20)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(set(ids) <= set(split["test"]))
        self.assertTrue(set(ids).isdisjoint(split["train"]))
        papers = load_sample().set_index("id")
        for item in labels:
            self.assertIsInstance(item["human_verified"], bool)
            self.assertTrue(item["labelled"])
            self.assertEqual(item["title"].strip(), papers.loc[item["id"], "title"].strip())
            self.assertEqual(item["abstract"].strip(), papers.loc[item["id"], "abstract"].strip())
            abstract = item["abstract"].casefold()
            for field in ("methods", "datasets"):
                for name in item[field]:
                    with self.subTest(paper=item["id"], field=field, name=name):
                        self.assertIn(name.casefold(), abstract)
