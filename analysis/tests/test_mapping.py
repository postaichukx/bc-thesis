import unittest

import pandas as pd

from bc_analysis.config import KNOWN_LOCAL_FEATURES
from bc_analysis.mapping import _holdout_ids, deanonymize_rows, fit_calibrations


class MappingTests(unittest.TestCase):
    def test_holdout_is_reproducible_and_does_not_mutate_input(self):
        frame = pd.DataFrame(
            {
                "txId": list(range(1, 61)),
                "timestep": [20] * 10
                + [29] * 10
                + [30] * 10
                + [32] * 10
                + [35] * 10
                + [42] * 10,
            }
        )
        original = frame.copy(deep=True)
        first = _holdout_ids(frame, 42)
        second = _holdout_ids(frame, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        pd.testing.assert_frame_equal(frame, original)

    def test_affine_calibration_recovers_known_values(self):
        elliptic_rows = []
        raw_rows = []
        for offset, txid in enumerate(("a", "b", "c")):
            normalized = float(offset - 1)
            elliptic = {"txId": offset + 1, "timestep": 20, "txid": txid}
            raw = {"txid": txid}
            for feature_number, name in KNOWN_LOCAL_FEATURES.items():
                elliptic[f"feature_{feature_number}"] = normalized
                raw[name] = 10 * normalized + 20
            elliptic_rows.append(elliptic)
            raw_rows.append(raw)

        calibrations = fit_calibrations(pd.DataFrame(elliptic_rows), pd.DataFrame(raw_rows))
        predicted = deanonymize_rows(pd.DataFrame([elliptic_rows[2]]), calibrations)
        for name in KNOWN_LOCAL_FEATURES.values():
            self.assertAlmostEqual(float(predicted.iloc[0][name]), 30.0)


if __name__ == "__main__":
    unittest.main()
