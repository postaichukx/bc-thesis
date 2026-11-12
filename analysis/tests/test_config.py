import unittest

from bc_analysis.config import BLOCK_RANGES, SELECTED_TIMESTEPS, TIMESTEP_SPLIT


class ConfigurationTests(unittest.TestCase):
    def test_selected_time_steps_have_ordered_non_overlapping_block_ranges(self):
        previous_end = -1
        for time_step in SELECTED_TIMESTEPS:
            start, end = BLOCK_RANGES[time_step]
            self.assertLessEqual(start, end)
            self.assertGreater(start, previous_end)
            previous_end = end

    def test_selected_time_steps_cover_train_validation_and_test(self):
        self.assertEqual(set(SELECTED_TIMESTEPS), set(TIMESTEP_SPLIT))
        self.assertEqual(
            {split: list(TIMESTEP_SPLIT.values()).count(split) for split in set(TIMESTEP_SPLIT.values())},
            {"train": 2, "validation": 2, "test": 2},
        )


if __name__ == "__main__":
    unittest.main()
