# data_pipeline.py (PROJECT ROOT)

from pathlib import Path
import sys

# Make src importable
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from config import Settings
from data.fetch import MarketDataClient, ingest_bronze
from data.clean import run_clean_pipeline


def main():
    s = Settings()

    client = MarketDataClient(
        base_url=s.base_url,
        token=s.token,
    )

    RAW_DIR = ROOT / "data" / "spy_option" / "raw"
    PROCESSED_DIR = ROOT / "data" / "spy_option" / "processed"
    OUT_CSV = PROCESSED_DIR / "market_data.csv"
    VIX_CSV = PROCESSED_DIR / "vix.csv"

    ingest_bronze(
        client=client,
        out_dir=RAW_DIR,
        symbol="SPY",
        start_date="2026-02-01",
        end_date="2026-02-03",
        side="call",
        sleep_sec=2.0,
    )

    run_clean_pipeline(
        raw_dir=RAW_DIR,
        out_csv=OUT_CSV,
        vix_csv=VIX_CSV,
    )


if __name__ == "__main__":
    main()
