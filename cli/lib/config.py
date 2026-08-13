from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = PROJECT_ROOT / "cache"

DATA_PATH: Path = DATA_DIR / "movies.json"
STOPWORDS_PATH: Path = DATA_DIR / "stopwords.txt"

INDEX_CACHE_PATH: Path = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_PATH: Path = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_CACHE_PATH: Path = CACHE_DIR / "term_frequencies.pkl"
DOC_LENGTHS_CACHE_PATH: Path = CACHE_DIR / "doc_lengths.pkl"

EMBEDDINGS_CACHE_PATH: Path = CACHE_DIR / "embeddings.npy"
EMBEDDINGS_META_CACHE_PATH: Path = CACHE_DIR / "embeddings.meta.json"
CHUNK_EMBEDDINGS_CACHE_PATH: Path = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_CACHE_PATH: Path = CACHE_DIR / "chunk_metadata.json"

DEFAULT_SEARCH_LIMIT: int = 5

BM25_K1: float = 1.5
BM25_B: float = 0.75

SEMANTIC_CANDIDATE_MULTIPLIER: int = 500
RERANK_FETCH_MULTIPLIER: int = 5

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-TinyBERT-L2-v2"

LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_API_KEY_ENV: str = "OPENROUTER_API_KEY"
LLM_DEFAULT_MODEL: str = "google/gemma-4-26b-a4b-it:free"
LLM_RERANK_MODEL: str = "google/gemma-4-26b-a4b-it:free"
LLM_TIMEOUT: float = 60.0
LLM_MAX_RETRIES: int = 2
