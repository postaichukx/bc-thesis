import unittest

import numpy as np

from bc_analysis.kap_05_experiment.model_training import _scale_from_training


class ModelPreparationTests(unittest.TestCase):
    def test_scaler_uses_training_rows_only(self):
        features = np.array([[0.0], [2.0], [100.0]])
        train_scope = np.array([True, True, False])

        scaled = _scale_from_training(features, train_scope)

        self.assertAlmostEqual(float(scaled[0, 0]), -1.0)
        self.assertAlmostEqual(float(scaled[1, 0]), 1.0)
        self.assertAlmostEqual(float(scaled[2, 0]), 99.0)

    def test_scaler_rejects_empty_training_scope(self):
        with self.assertRaises(ValueError):
            _scale_from_training(np.array([[1.0]]), np.array([False]))


if __name__ == "__main__":
    unittest.main()
