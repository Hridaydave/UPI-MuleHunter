from dataclasses import dataclass
import os

@dataclass
class Settings:
    api_host: str = os.getenv("MULEHUNTER_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("MULEHUNTER_PORT", "8000"))
    results_dir: str = os.getenv("MULEHUNTER_RESULTS", "results")
    data_dir: str = os.getenv("MULEHUNTER_DATA", "data")

settings = Settings()
