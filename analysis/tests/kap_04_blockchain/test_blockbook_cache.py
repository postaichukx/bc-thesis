import tempfile
import unittest
from pathlib import Path

from bc_analysis.kap_04_blockchain.blockbook_client import BlockbookCache


class CacheTests(unittest.TestCase):
    def test_round_trip_and_non_success_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = BlockbookCache(Path(directory) / "cache.sqlite3")
            cache.put("ok", "https://example.test/ok", 200, {"value": 7})
            cache.put("bad", "https://example.test/bad", 500, {"error": "temporary"})
            self.assertEqual(cache.get("ok"), {"value": 7})
            self.assertIsNone(cache.get("bad"))


if __name__ == "__main__":
    unittest.main()
