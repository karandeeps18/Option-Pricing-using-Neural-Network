import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = list(Path(__file__).parents)[2]
ENV_DIR = BASE_DIR / ".env"

load_dotenv()

class Settings:
    def __init__(self):
        self.base_url = "https://api.marketdata.app/v1"
        self.token = os.getenv("MARKETDATA_TOKEN")
