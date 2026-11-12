from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import DEFAULT_USER_AGENT


class BlockbookCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    status INTEGER NOT NULL,
                    body TEXT NOT NULL,
                    fetched_at INTEGER NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def get(self, key: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT body FROM responses WHERE cache_key = ? AND status = 200", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, url: str, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO responses(cache_key, url, status, body, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    url=excluded.url,
                    status=excluded.status,
                    body=excluded.body,
                    fetched_at=excluded.fetched_at
                """,
                (key, url, status, body, int(time.time())),
            )


class BlockbookClient:
    def __init__(
        self,
        base_url: str,
        cache: BlockbookCache,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 30.0,
        retries: int = 4,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries

    def get_json(self, path: str, params: dict[str, object] | None = None) -> dict:
        query = urllib.parse.urlencode(params or {})
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        cache_key = url
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    self.cache.put(cache_key, url, response.status, payload)
                    return payload
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
                last_error = error
                status = getattr(error, "code", None)
                if status in {400, 404}:
                    break
                if attempt < self.retries:
                    time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"Blockbook request failed: {url}: {last_error}")

    def transaction(self, txid: str) -> dict:
        return self.get_json(f"/api/v2/tx/{txid}")

    def block_page(self, height: int, page: int = 1, page_size: int = 1000) -> dict:
        return self.get_json(
            f"/api/v2/block/{height}",
            {"page": page, "pageSize": page_size, "details": "txs"},
        )

    def block_transactions(self, height: int) -> list[dict]:
        first = self.block_page(height, 1)
        transactions = list(first.get("txs") or first.get("transactions") or [])
        total_pages = int(first.get("totalPages") or 1)
        for page in range(2, total_pages + 1):
            payload = self.block_page(height, page)
            transactions.extend(payload.get("txs") or payload.get("transactions") or [])

        result: list[dict] = []
        for transaction in transactions:
            if isinstance(transaction, dict):
                result.append(transaction)
            else:
                result.append(self.transaction(str(transaction)))
        return result


def fetch_transactions(
    client: BlockbookClient, txids: list[str], workers: int = 4
) -> tuple[list[dict], list[tuple[str, str]]]:
    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(client.transaction, txid): txid for txid in txids}
        for future in as_completed(futures):
            txid = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # recorded for resumable collection
                failures.append((txid, str(error)))
    return results, failures
