import unittest
from niche_screen.demo import LEXICON, synthetic_data
from niche_screen.evaluation import f1, site_bootstrap
from niche_screen.model import SharedIntentScreen


class ToolkitTest(unittest.TestCase):
    def test_f1_and_unassessed(self):
        self.assertAlmostEqual(f1([1, 1, 0, None], [1, 0, 1, 1]), .5)
        with self.assertRaises(ValueError):
            f1([1], [])

    def test_bootstrap_preserves_sites_and_is_deterministic(self):
        rows = [{"site": "a", "label": 1, "prediction": 1},
                {"site": "a", "label": 0, "prediction": 1},
                {"site": "b", "label": 1, "prediction": 0}]
        result = site_bootstrap(rows, 100, seed=1)
        self.assertEqual(result, site_bootstrap(rows, 100, seed=1))
        self.assertEqual(result["sites"], 2)
        self.assertAlmostEqual(result["f1"], .5)
        self.assertLessEqual(result["ci95"][0], result["ci95"][1])

    def test_end_to_end_and_input_checks(self):
        pages, labels = synthetic_data()
        train = [p for p in pages if int(p["site"][-2:]) < 30]
        held = [p for p in pages if int(p["site"][-2:]) >= 30]
        model = SharedIntentScreen(LEXICON).fit_pages(train).fit_operators(train, labels)
        predictions = model.predict_sites(held)
        self.assertEqual(len(predictions), 10)
        for topics in predictions.values():
            for result in topics.values():
                self.assertTrue(0 <= result["page_score"] <= 1)
                self.assertTrue(0 <= result["operator_score"] <= 1)
                self.assertIsInstance(result["commercial"], bool)
        with self.assertRaises(ValueError):
            model.site_features(held[:4])
        bad = dict(held[0], embedding=[0])
        with self.assertRaises(ValueError):
            model.page_scores([bad])


if __name__ == "__main__":
    unittest.main()

class OwnListsTest(unittest.TestCase):
    def test_exact_destination_labels(self):
        from pathlib import Path
        from niche_screen.inputs import load_lists, categorize_links
        cues, destinations = load_lists(Path(__file__).resolve().parents[1] / 'examples/own-lists.json')
        self.assertIn('gaming', cues)
        self.assertEqual(categorize_links(['https://example.test/path', 'https://other.example.test'], destinations), ['gaming'])
