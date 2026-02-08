import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path.cwd().parent  # if in Notebooks/
PROCESSED = ROOT / "data" / "spy_option" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

vix = yf.download("^VIX", start="2024-12-30", end="2026-02-05")["Close"] / 100
vix = vix.shift(1)  # lag
vix = vix.reset_index().rename(columns={"Date": "snapshot_date", "Close": "VIX"})

vix.to_csv(PROCESSED / "vix.csv", index=False)
