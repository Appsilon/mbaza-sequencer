from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    csv_path: Path
    image_path: Path
    max_seq: float
    max_delay: float
