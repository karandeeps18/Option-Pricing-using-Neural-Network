# src/data/clean.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# SPOT FROM THE 'C' close 
def _extract_spot_close(spot_payload: Dict[str, Any]) -> Optional[float]:
    c = spot_payload.get("c")
    if c is None:
        return None
    if isinstance(c, list):
        return float(c[0]) if len(c) else None
    if isinstance(c, (int, float)):
        return float(c)
    return None


# load the bronze data as json 
def load_bronze(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No bronze json files in {raw_dir}")

    frames: List[pd.DataFrame] = []
    for fp in files:
        with open(fp, "r") as f:
            raw = json.load(f)

        chain = raw.get("chain", {})
        df = pd.DataFrame(chain)

        df["snapshot_date"] = raw.get("snapshot_date")
        df["spot_price"] = _extract_spot_close(raw.get("spot", {}))

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


# build features M, T_ann
def build_features(final_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "optionSymbol", "expiration", "dte",
        "intrinsicValue", "extrinsicValue",
        "strike", "mid", "underlyingPrice",
        "spot_price", "snapshot_date",
        "volume", "inTheMoney",
    ]
    missing = [c for c in cols if c not in final_df.columns]
    if missing:
        raise KeyError(f"Missing columns from chain payload: {missing}")

    df = final_df[cols].copy()

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["strike"] = df["strike"].astype(float)
    df["mid"] = df["mid"].astype(float)
    df["underlyingPrice"] = df["underlyingPrice"].astype(float)
    df["dte"] = df["dte"].astype(float)

    # maturity in YEARS (using calendar)
    df["t_ann"] = df["dte"] / 365.0

    # log-moneyness
    df["M"] = np.log(df["underlyingPrice"] / df["strike"])

    return df


# using VIX for the sigma 
def merge_vix_csv(df: pd.DataFrame, vix_csv: Path) -> pd.DataFrame:
    vix = pd.read_csv(vix_csv)
    vix["snapshot_date"] = pd.to_datetime(vix["snapshot_date"])
    if "VIX" not in vix.columns:
        raise KeyError("vix_csv must contain a 'VIX' column")
    return df.merge(vix[["snapshot_date", "VIX"]], on="snapshot_date", how="left")


# featur constrain and target C/K 
def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    # hard constraints
    df = df[df["t_ann"] > 0].copy()                                   # time to maturity > 0 
    df = df[(df["strike"] > 0) & (df["mid"] >= 0)].copy()             # strike > 0 and mid >= 0

    # finite inputs
    Xcols = ["M", "t_ann"]
    if "VIX" in df.columns and df["VIX"].notna().any():
        Xcols.append("VIX")
                            

    # targets
    df["C"] = df["mid"]                                                # taking mid as the target 
    df["C/K"] = df["mid"] / df["strike"]                              # normalized target 

    # uniqueness
    assert not df.duplicated(["optionSymbol", "snapshot_date"]).any(), "Duplicate contract/day rows"
    return df

# Check if vix exist for the date and build the clean pippeline 
def run_clean_pipeline( *, raw_dir: Path,out_csv: Path, vix_csv: Optional[Path] = None,) -> pd.DataFrame:
    final_df = load_bronze(raw_dir)
    feats = build_features(final_df)

    # merge 
    if vix_csv is not None and Path(vix_csv).exists():
        feats = merge_vix_csv(feats, vix_csv)
    else:
        feats["VIX"] = np.nan
        if vix_csv is not None:
            print(f"[WARN] vix.csv not found at {vix_csv}. Setting VIX=NaN.")


    feats["VIX"] = feats["VIX"].astype(float)

    cleaned = apply_quality_filters(feats)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} rows={len(cleaned):,}")
    return cleaned
