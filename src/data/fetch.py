# src/data/fetch.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import requests


def build_chain_url(
    base_url: str,
    symbol: str,
    *,
    date: Optional[str] = None,
    expiration: Optional[str] = None,
    side: Optional[str] = None,
    columns: Optional[str] = None,
) -> str:
    params = []
    for k, v in {
        "date": date,
        "expiration": expiration,
        "side": side,
        "columns": columns,
    }.items():
        if v is not None:
            params.append(f"{k}={v}")
    qs = "&".join(params)
    return f"{base_url}/options/chain/{symbol}/?{qs}" if qs else f"{base_url}/options/chain/{symbol}/"


def build_spot_url(
    base_url: str,
    resolution: str,
    symbol: str,
    *,
    date: Optional[str] = None,
    columns: Optional[str] = None,
) -> str:
    params = []
    for k, v in {"date": date, "columns": columns}.items():
        if v is not None:
            params.append(f"{k}={v}")
    qs = "&".join(params)
    # IMPORTANT: single slash
    return f"{base_url}/stocks/candles/{resolution}/{symbol}/?{qs}" if qs else f"{base_url}/stocks/candles/{resolution}/{symbol}/"


@dataclass(frozen=True)
class MarketDataClient:
    base_url: str
    token: str
    timeout: int = 60
    max_retries: int = 3
    backoff_sec: float = 2.0

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Token {self.token}", "Accept": "application/json"}

    def get_json(self, url: str) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.get(url, headers=self.headers, timeout=self.timeout)
                if r.status_code == 404:
                    return {"_status_code": 404}
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_sec * attempt)
                else:
                    raise RuntimeError(f"GET failed ({url}): {last_exc}") from last_exc
        raise RuntimeError("Unreachable")


def iter_trading_days(start: str, end: str) -> Iterable[str]:
    import pandas as pd
    for day in pd.date_range(start=start, end=end, freq="B"):
        yield day.strftime("%Y-%m-%d")


def ingest_bronze(
    *,
    client: MarketDataClient,
    out_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    side: str = "call",
    spot_resolution: str = "D",
    sleep_sec: float = 1.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for query_date in iter_trading_days(start_date, end_date):
        fp = out_dir / f"{symbol.lower()}_{query_date}.json"
        if fp.exists():
            print(f"skip {query_date}: exists")
            continue

        chain_url = build_chain_url(client.base_url, symbol, date=query_date, side=side)
        chain = client.get_json(chain_url)
        if chain.get("_status_code") == 404:
            print(f"skip {query_date}: no chain (holiday/closed)")
            continue

        spot_url = build_spot_url(client.base_url, spot_resolution, symbol, date=query_date)
        spot = client.get_json(spot_url)
        if spot.get("_status_code") == 404:
            print(f"skip {query_date}: no spot (holiday/closed)")
            continue

        payload = {
            "snapshot_date": query_date,
            "symbol": symbol,
            "side": side,
            "chain": chain,
            "spot": spot,
        }

        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w") as f:
            json.dump(payload, f)

        print(f"Bronze ingested: {query_date}")
        time.sleep(sleep_sec)
