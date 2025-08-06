import os
from dotenv import load_dotenv

load_dotenv()

class Config:
  """
    Configuration class for the application.
    This class loads environment variables from a .env file and provides access to them.
  """
  
  FLASK_APP = os.getenv("FLASK_APP", "app.py")
  FLASK_RUN_PORT = int(os.getenv("FLASK_RUN_PORT", 5000))
  FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
  CHROMA_PERSISTENCE_DIR = os.getenv("CHROMA_PERSISTENCE_DIR", "../../data/vector_databases/chroma_db_raw_boardgames")
  CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "raw_boardgames")
  OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
  OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")