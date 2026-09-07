from __future__ import annotations
import os
from pathlib import Path


#Define the base directory and various data directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "fact_layer.db"



### Ensure that the data directories exist
for _dir in (DATA_DIR,UPLOAD_DIR,CACHE_DIR):
    _dir.mkdir(parents=True,exist_ok=True)
    


### Define constants for text processing
LOW_TEXT_CHAR_THRESHOLD = 40   # a page under this many chars is likely image-only
CHUNK_TARGET_CHARS = 3500
CHUNK_OVERLAP_CHARS = 500

### Define constants for LLM providers and API keys
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")