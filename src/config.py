"""
Configuration module for Multi-Agent RAG system.
Manages API keys, model settings, and application defaults.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── API Configuration ───
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def is_api_available():
    """Check if a valid Gemini API key is configured."""
    return bool(GEMINI_API_KEY) and GEMINI_API_KEY != "your_api_key_here"

# ─── Paths ───
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ─── Embedding Model ───
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# ─── Chunking ───
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks

# ─── Retrieval ───
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# ─── LLM ───
GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.3
MAX_TEMPERATURE = 1.0

# ─── Hallucination Thresholds ───
DEFAULT_HALLUCINATION_THRESHOLD = 0.4   # risk above this → reject/refine
DEFAULT_EVIDENCE_THRESHOLD = 0.35       # min similarity to consider relevant
DEFAULT_MAX_REGENERATION_ATTEMPTS = 2

# ─── Mode Labels ───
OFFLINE_MODE_LABEL = "🔌 Offline Mode"
LIVE_MODE_LABEL = "🟢 Live Gemini Mode"
