import unittest
from pathlib import Path
import numpy as np
from niche_screen.inference import load_model


class FrozenInferenceTest(unittest.TestCase):
    def test_real_weights_score_synthetic_inputs(self):
        model = load_model(Path(__file__).resolve().parents[1] / 'weights')
        rng = np.random.default_rng(27)
        v = rng.normal(size=(3, 1024)).astype('f')
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        a7 = np.zeros((3, 9, 19), dtype='f')
        op = np.zeros((3, 9, 16), dtype='f')
        pages = model.score_pages(v, a7)
        site = model.score_site(v, a7, op, [True, False, False])
        self.assertEqual(pages['commercial_score'].shape, (3, 9))
        self.assertEqual(site['operator_score'].shape, (9,))
        np.testing.assert_allclose(site['commercial_score'], pages['commercial_score'].max(axis=0))
        self.assertTrue(np.isfinite(site['operator_score']).all())

    def test_missing_vectors_are_not_negatives(self):
        model = load_model(Path(__file__).resolve().parents[1] / 'weights')
        with self.assertRaises(ValueError):
            model.score_pages(np.zeros((1, 1024)), np.zeros((1, 9, 19)))

class RetrainTest(unittest.TestCase):
    def test_fit_exports_reloadable_weights(self):
        import tempfile
        from niche_screen.retrain import fit, synthetic
        data = synthetic()
        with tempfile.TemporaryDirectory() as directory:
            result = fit(data, directory, Path(__file__).resolve().parents[1] / 'weights')
            model = load_model(directory)
            prediction = model.score_site(data['vectors'][:3], data['page_features'][:3],
                                          data['operator_features'][:3], data['homepage'][:3])
            self.assertTrue(np.isfinite(prediction['operator_score']).all())
            self.assertEqual(result['sites'], 10)
