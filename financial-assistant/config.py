import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = DATA_DIR / "faiss_index"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"

for d in [DATA_DIR, MODELS_DIR, INDEX_DIR, SAMPLE_DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/flan-t5-base")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
TOP_K_GENERATION = int(os.getenv("TOP_K_GENERATION", "50"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "FlatIP")
FAISS_NPROBE = int(os.getenv("FAISS_NPROBE", "10"))
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

PROMPT_TEMPLATES = {
    "default": (
        "Answer the following financial question using only the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "chain_of_thought": (
        "You are a financial analyst. Use the provided context to answer the question.\n"
        "Think step by step and cite specific data points.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Step-by-step analysis and answer:"
    ),
    "concise": (
        "Based on the context below, provide a brief and precise answer.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Brief answer:"
    ),
}
