import math
import unittest

from bc_analysis.kap_04_blockchain.transaction_features import _as_satoshi, reconstruct_edges, transaction_features


class TransactionFeatureTests(unittest.TestCase):
    def test_amount_conversion_handles_satoshi_and_decimal_btc(self):
        self.assertEqual(_as_satoshi("1000"), 1000)
        self.assertEqual(_as_satoshi("0.00001000"), 1000)
        self.assertEqual(_as_satoshi("1.25"), 125_000_000)

    def test_fee_values_addresses_and_counts(self):
        transaction = {
            "txid": "current",
            "blockHeight": 100,
            "value": "99000000",
            "valueIn": "100000000",
            "fees": "1000000",
            "vin": [
                {"txid": "old-a", "value": "60000000", "addresses": ["A"]},
                {"txid": "old-b", "value": "40000000", "addresses": ["B"]},
            ],
            "vout": [
                {"value": "80000000", "addresses": ["C"]},
                {"value": "19000000", "addresses": ["A"]},
            ],
        }
        result = transaction_features(transaction)
        self.assertEqual(result["value_satoshi"], 99_000_000)
        self.assertEqual(result["fee_satoshi"], 1_000_000)
        self.assertEqual(result["input_count"], 2)
        self.assertEqual(result["output_count"], 2)
        self.assertEqual(result["unique_input_addresses"], 2)
        self.assertEqual(result["unique_output_addresses"], 2)
        self.assertEqual(result["predecessor_txids"], "old-a|old-b")
        self.assertTrue(math.isclose(result["input_value_mean"], 50_000_000))

    def test_coinbase_without_inputs_has_zero_input_statistics(self):
        result = transaction_features(
            {"txid": "coinbase", "value": "50", "vin": [], "vout": [{"value": "50"}]}
        )
        self.assertEqual(result["input_count"], 0)
        self.assertEqual(result["input_value_mean"], 0.0)
        self.assertEqual(result["fee_satoshi"], 0)

    def test_reconstructed_edges_use_previous_transaction_hashes(self):
        rows = [
            {"txId": 1, "txid": "a", "predecessor_txids": ""},
            {"txId": 2, "txid": "b", "predecessor_txids": "a|outside"},
            {"txId": 3, "txid": "c", "predecessor_txids": "b"},
        ]
        self.assertEqual(reconstruct_edges(rows), {(1, 2), (2, 3)})


if __name__ == "__main__":
    unittest.main()
