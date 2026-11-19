import os 
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL" , "openai/gpt-oss-20b")

PDF_DIR = BASE_DIR / "data" / "pdfs"
DB_DIR = BASE_DIR / "chroma_db"

# Configuración RAG
COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Configuración OCR (fallback para PDFs sin texto legible)
OCR_LANG = os.getenv("OCR_LANG", "spa")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "30"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "50"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))