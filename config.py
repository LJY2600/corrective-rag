from typing import Final

# --- Document & Ingestion ---
DEFAULT_DOC_URL: Final[str] = "https://arxiv.org/pdf/2307.09288.pdf"
CHUNK_SIZE: Final[int] = 500
CHUNK_OVERLAP: Final[int] = 100

# --- Embedding ---
EMBEDDING_MODEL: Final[str] = "text-embedding-v3"
EMBEDDING_DIMENSION: Final[int] = 1024  #阿里云 text-embedding-v3 输出维度

# --- LLM ---
LLM_MODEL: Final[str] = "qwen-flash"
LLM_BASE_URL: Final[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- Vector Store ---
CHROMA_PERSIST_DIR: Final[str] = "./chroma_db"

# --- Search ---
TAVILY_MAX_RESULTS: Final[int] = 3
TAVILY_SEARCH_DEPTH: Final[str] = "advanced"
