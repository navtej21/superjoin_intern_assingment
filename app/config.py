"""Central paths and tunables.

Nothing document-specific lives here — thresholds are generic properties of
PDF text extraction and chunking, not facts about any particular filing.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "fact_layer.db"

for _dir in (DATA_DIR, UPLOAD_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# A page below this many extracted characters is treated as "low text" —
# most likely a scanned/image-only page. We don't OCR it; we flag it.
LOW_TEXT_CHAR_THRESHOLD = 40

# Chunker targets, in characters (not tokens) — see app/chunker.py for why.
CHUNK_TARGET_CHARS = 3500
CHUNK_OVERLAP_CHARS = 500

# LLM provider selection for later phases. Not used yet.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")