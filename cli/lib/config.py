from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = PROJECT_ROOT / "cache"

DATA_PATH: Path = DATA_DIR / "movies.json"
GOLDEN_DATASET_PATH: Path = DATA_DIR / "golden_dataset.json"
STOPWORDS_PATH: Path = DATA_DIR / "stopwords.txt"

INDEX_CACHE_PATH: Path = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_PATH: Path = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_CACHE_PATH: Path = CACHE_DIR / "term_frequencies.pkl"
DOC_LENGTHS_CACHE_PATH: Path = CACHE_DIR / "doc_lengths.pkl"

EMBEDDINGS_CACHE_PATH: Path = CACHE_DIR / "embeddings.npy"
CHUNK_EMBEDDINGS_CACHE_PATH: Path = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_CACHE_PATH: Path = CACHE_DIR / "chunk_metadata.json"

DEFAULT_SEARCH_LIMIT: int = 5

BM25_K1: float = 1.5
BM25_B: float = 0.75

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-TinyBERT-L2-v2"

LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_API_KEY_ENV: str = "OPENROUTER_API_KEY"
LLM_DEFAULT_MODEL: str = "openrouter/free"
LLM_RERANK_MODEL: str = "google/gemma-4-26b-a4b-it:free"
