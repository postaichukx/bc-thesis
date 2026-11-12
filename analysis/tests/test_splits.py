import unittest

import pandas as pd

from bc_analysis.datasets import _clean_split, filter_edges_for_split


class SplitTests(unittest.TestCase):
    def test_clean_split_drops_later_transactions_with_shared_address(self):
        nodes = pd.DataFrame(
            [
                {
                    "txId": 1,
                    "split_temporal": "train",
                    "transparent_available": True,
                    "input_addresses": "A",
                    "output_addresses": "B",
                },
                {
                    "txId": 2,
                    "split_temporal": "train",
                    "transparent_available": True,
                    "input_addresses": "B",
                    "output_addresses": "C",
                },
                {
                    "txId": 3,
                    "split_temporal": "validation",
                    "transparent_available": True,
                    "input_addresses": "C",
                    "output_addresses": "E",
                },
                {
                    "txId": 4,
                    "split_temporal": "test",
                    "transparent_available": True,
                    "input_addresses": "D",
                    "output_addresses": "F",
                },
            ]
        )
        self.assertEqual(
            _clean_split(nodes).tolist(), ["train", "train", "dropped", "test"]
        )

    def test_clean_graph_removes_cross_split_edges(self):
        nodes = pd.DataFrame(
            {"txId": [1, 2, 3], "split_clean": ["train", "test", "dropped"]}
        )
        edges = pd.DataFrame({"txId1": [1, 1, 2], "txId2": [1, 2, 3]})
        result = filter_edges_for_split(edges, nodes, "split_clean")
        self.assertEqual(result[["txId1", "txId2"]].values.tolist(), [[1, 1]])


if __name__ == "__main__":
    unittest.main()
