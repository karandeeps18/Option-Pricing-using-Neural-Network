# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (adjust if needed)
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

# Explicitly load .env
load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    def __init__(self):
        self.base_url = "https://api.marketdata.app/v1"
        self.token = os.getenv("MARKETDATA_TOKEN")

        if not self.token:
            raise RuntimeError(
                "MARKETDATA_TOKEN not found. "
                "Check that .env exists at project root and is loaded."
            )

settings = Settings()
