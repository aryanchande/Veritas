"""All tunables in one place. Override with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
FRONTEND_DIR = ROOT / "frontend"

# --- Ollama (can run on another machine, e.g. OLLAMA_HOST=http://192.168.1.20:11434) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")      # use qwen3.5:4b on an 8 GB machine
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")       # multilingual, handles Hindi
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "900"))

# --- retrieval / verification ---
TOP_K = int(os.getenv("TOP_K", "8"))
MAX_EVIDENCE = int(os.getenv("MAX_EVIDENCE", "12"))
MIN_SIM = float(os.getenv("MIN_SIM", "0.35"))          # best cosine below this -> abstain before calling the LLM
QUOTE_MIN_RATIO = float(os.getenv("QUOTE_MIN_RATIO", "0.9"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1000"))
SCAN_SIM = float(os.getenv("SCAN_SIM", "0.60"))        # similarity for candidate conflict groups
SCAN_MAX_GROUPS = int(os.getenv("SCAN_MAX_GROUPS", "60"))
