# RAG Search Engine

A hands-on, from-scratch learning project that builds a full **Retrieval-Augmented Generation (RAG)** search engine for a catalog of **5,000 movies** (Webflyx). Every layer of the stack was implemented incrementally and deliberately — starting from a plain keyword index all the way to hybrid semantic retrieval, LLM reranking, answer generation, and multimodal (image) search.

This README documents **everything done in the project**, framed as a learning journey. Follow the git history ([`git log --oneline`](#learning-journey)) to see the exact order each concept was built.

---

## The Big Picture

```
                     ┌──────────────────────────────────────────────┐
                     │                QUERY (text / image)          │
                     └──────────────┬───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │          Query Processing (optional)          │
                    │   Spell-fix · Rewrite · Expand (LLM)          │
                    └───────────────┬───────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────────┐
        │            RETRIEVAL                                      │
        │  ┌──────────────┐     ┌───────────────────┐               │
        │  │  BM25 / TF-IDF│     │  Dense Semantics   │              │
        │  │ Inverted Index│     │  Chunk embeddings  │              │
        │  │ (keyword)     │     │  + FAISS (ANN)     │              │
        │  └──────┬───────┘     └─────────┬─────────┘               │
        │         └──────────┬────────────┘                         │
        │              Hybrid Fusion                                 │
        │       · Weighted (α·BM25 + (1-α)·semantic)                │
        │       · Reciprocal Rank Fusion (RRF)                      │
        └───────────────┬───────────────────────────────────────────┘
                        │
        ┌───────────────▼───────────────────────────────────────────┐
        │                  RE-RANKING (optional)                    │
        │   Individual LLM · Batch LLM · Cross-Encoder              │
        └───────────────┬───────────────────────────────────────────┘
                        │
        ┌───────────────▼───────────────────────────────────────────┐
        │                  GENERATION (RAG)                         │
        │   Answer · Summarize · Answer w/ citations · Q&A          │
        └───────────────────────────────────────────────────────────┘
```

Everything is piped through **cached artifacts** (pickle/numpy/json/FAISS index) so slow steps — embeddings, indexes — are computed once and reused.

---

## Learning Journey

The project evolved through 54 commits. Each phase below maps to those commits.

### 1. Project Setup & CLI Foundation
- Package scaffolded with `uv` / `pyproject.toml` (Python 3.13).
- A `cli/` package with thin `argparse`-based entry points for every tool built in the project.
- Data: `data/movies.json` (5,000 movies with `id`, `title`, `description`) and `data/stopwords.txt` (198 stopwords).

### 2. Keyword Search (the "database" way first)
Built a classic **inverted index** from scratch and a family of scoring functions — this taught the fundamentals before moving to neural search:

- **Tokenizer pipeline** in `cli/lib/search_utils.py`: lowercase → strip punctuation → remove stopwords → **Porter stemming** (via NLTK). Includes a deterministic, order-preserving deduplication so results don't vary with `PYTHONHASHSEED`.
- **Inverted index** (`cli/lib/inverted_index.py`): `term → [doc_id]` postings with per-document term frequencies, doc lengths, and document mapping.
- **TF, IDF, TF-IDF** scoring with dirty CSV-style CLI commands (`tf`, `idf`, `tfidf`).
- **BM25**: proper IDF form `log((N - df + 0.5)/(df + 0.5) + 1)`, saturated TF with tunable `k1` / `b` length normalization, and a full `bm25search` end-to-end command.
- **Persistence**: the index is serialized to `cache/` and rebuilt only when missing.

### 3. Semantic Search (embeddings + ANN)
- **Sentence embeddings** with `all-MiniLM-L6-v2` (384-dim) via `sentence-transformers`; every document is encoded once and cached to `cache/embeddings.npy` with a **SHA-256 fingerprint** of the source data to invalidate stale caches.
- **Semantic (sentence) chunking** (`cli/lib/chunked_semantic_search.py`): split descriptions at sentence boundaries, chunk with configurable `max_chunk_size` and **overlap**, embed each chunk, and aggregate chunk scores back to whole movies.
- **Why chunk?** — documents were long; whole-doc embeddings diluted the signal. Chunking gave finer matches that could be rolled up per document.
- **FAISS ANN index** (`cli/lib/faiss_index.py`): `IndexFlatIP` over L2-normalized embeddings so inner-product == cosine similarity. Cached to `cache/faiss.index` / `cache/faiss_chunk.index`; search subclasses handle individual vs. chunked retrieval.

### 4. Hybrid Search (best of both worlds)
Merged lexical + semantic retrieval in `cli/lib/hybrid_search.py`:

- **Weighted fusion**: min-max normalize BM25 and semantic scores, then combine with a tunable `α` (`α·bm25 + (1−α)·semantic`).
- **Reciprocal Rank Fusion (RRF)**: `Σ 1/(k + rank)` per list, `k = 60`. Rank-based fusion — no score calibration needed.

### 5. LLM Query Enhancement
Using OpenRouter (`openrouter/free`) through an OpenAI-compatible client, three prompt-based query transforms were added (`cli/lib/prompts.py`):

- **Spell-fix**: correct only high-confidence typos, preserve everything else.
- **Rewrite**: turn conversational queries ("that bear movie where leo gets attacked") into Google-style search queries.
- **Expand**: append synonyms / related concepts to broaden recall.

### 6. Re-ranking (getting precision back)
Because the pipeline fetches *more* results than requested (5×), re-ranking improves the top-k (`cli/lib/rerankers.py`):

- **Individual LLM reranker** — scores each candidate 0–10 via prompt, one call at a time (with delay for rate limits); falls back to RRF order on LLM failure.
- **Batch LLM reranker** — asks the LLM to return a raw JSON array of document IDs ordered by relevance.
- **Cross-encoder reranker** — `cross-encoder/ms-marco-TinyBERT-L2-v2` scores `(query, doc)` pairs locally (no API cost).

### 7. Retrieval-Augmented Generation (RAG)
`cli/augmented_generation_cli.py` wires retrieval → generation:

- `rag` — retrieve via RRF, then answer the query from retrieved docs.
- `summarize` — synthesize multiple search results into a dense multi-movie summary.
- `citations` — answer with `[1]`, `[2]`-style source citations; explicitly says "I don't have enough information" when appropriate.
- `question` — casual, conversational QA grounded in retrieved docs.

### 8. Evaluation (measuring retrieval quality)
- **Golden dataset** — `data/golden_dataset.json`: hand-built query → relevant-movie pairs.
- Retrieval evaluated against it (`cli/evaluation_cli.py`): **Precision@k**, **Recall@k**, and **F1@k** per query.
- Optional **LLM-based evaluation** (`--evaluate` flag in hybrid search): the model rates each retrieved result 0–3 for relevance.

### 9. Multimodal Search (images)
- **CLIP embeddings** (`clip-ViT-B-32`) project text *and* images into a shared space (`cli/lib/multimodal_search.py`).
- `verify_image_embedding` — inspect embedding shape.
- `image_search` — embed a query image and rank movies by cosine similarity (try `data/paddington.jpeg`).
- `describe_image_cli` — send image + text query to a vision-capable LLM to **rewrite the text query** from the visual content (base64 data-URL image input).

### 10. Engineering & Refactoring (the "senior" pass)
The final stretch of commits hardened the codebase:

- **Thin CLIs, fat lib**: moved all logic into `cli/lib/*`, kept `argparse` wrappers in `cli/*_cli.py`, purged presentation from the library layer.
- **Centralized configuration** (`cli/lib/config.py`): paths, models, hyperparameters (BM25 k1/b, RRF k, candidate multipliers) in one place.
- **Domain exceptions** (`cli/lib/exceptions.py`): `IndexNotFoundError`, `CacheInvalidError`, `EmptyQueryError`, `GenerationError`.
- **Result dataclasses** (`cli/lib/ranking.py`): `ScoredResult`, `WeightedResult`, `RRFResult` — frozen, comparable, deterministic.
- **Centralized cache serialization + index lifecycle** (`cli/lib/caches.py` + `faiss_index.py`): pickles, numpy, JSON, and FAISS index read/write with fingerprint-based invalidation.
- **Determinism**: seeded RNG (`models.seed_all`) so embeddings are reproducible on-device.
- **Test suite**: 42 tests across 8 test files covering caches, tokenization, inverted index, ranking math, chunking, FAISS, semantic search, and rerankers.

---

## Project Structure

```
├── cli/                         # thin argparse entry points
│   ├── keyword_search_cli.py    # tf, idf, tfidf, bm25tf, bm25idf, bm25search, search
│   ├── semantic_search_cli.py   # verify, embed_text, search, search_chunked, semantic_chunk, embed_chunks
│   ├── hybrid_search_cli.py     # normalize, weighted-search, rrf-search (+enhance/rerank/evaluate)
│   ├── augmented_generation_cli.py  # rag, summarize, citations, question
│   ├── evaluation_cli.py        # precision/recall/F1 against golden dataset
│   ├── multimodal_search_cli.py # verify_image_embedding, image_search
│   ├── describe_image_cli.py    # vision-LLM query rewriting from an image
│   └── lib/                     # the actual engine (all logic lives here)
│       ├── config.py            # paths, models, hyperparameters
│       ├── caches.py            # pickle / numpy / json / fingerprint helpers
│       ├── document.py          # Document protocol
│       ├── movies.py            # MovieDocument + loader
│       ├── search_utils.py      # stopwords, tokenization, stemming
│       ├── inverted_index.py    # postings, TF-IDF, BM25
│       ├── keyword_search.py    # keyword/BM25 commands
│       ├── models.py            # seeded embedder / cross-encoder singletons
│       ├── semantic_search.py   # whole-doc embeddings + search
│       ├── chunked_semantic_search.py  # sentence chunking + chunk search
│       ├── faiss_index.py       # IndexFlatIP build/search/save/load
│       ├── hybrid_search.py     # weighted + RRF fusion
│       ├── ranking.py           # result dataclasses, normalization, RRF math
│       ├── rerankers.py         # individual/batch LLM + cross-encoder
│       ├── prompts.py           # query enhancement + rerank prompts
│       ├── llm.py               # OpenAI-compatible LLM wrapper (OpenRouter)
│       ├── multimodal_search.py # CLIP image/text search
│       └── exceptions.py        # domain exceptions
├── data/
│   ├── movies.json              # 5,000 movies
│   ├── stopwords.txt            # 198 stopwords
│   ├── golden_dataset.json      # eval query → relevant movies
│   └── paddington.jpeg          # sample query image
├── cache/                       # generated artifacts (gitignored)
│   ├── index.pkl, docmap.pkl, term_frequencies.pkl, doc_lengths.pkl
│   ├── embeddings.npy, embeddings.meta.json, faiss.index
│   ├── chunk_embeddings.npy, chunk_metadata.json, faiss_chunk.index
├── tests/                       # 42 tests, 8 files
└── pyproject.toml
```

---

## Getting Started

**Requirements:** Python 3.13, `uv` (or pip).

```bash
uv sync                     # install dependencies
```

Set your OpenRouter key (used by LLM features — reranking, query enhancement, RAG):

```bash
# create a .env file (already gitignored) with:
OPENROUTER_API_KEY="sk-or-v1-..."
```

First run downloads the embedding / cross-encoder models and builds caches into `cache/`; later runs load from cache (embeddings are invalidated automatically when `data/movies.json` changes).

### Dependencies (`pyproject.toml`)
- `faiss-cpu` — ANN search
- `sentence-transformers` — `all-MiniLM-L6-v2` (embeddings), `clip-ViT-B-32` (multimodal), cross-encoder reranker
- `openai` — LLM client (OpenRouter-compatible API)
- `nltk` — Porter stemmer
- `numpy`, `pillow`, `python-dotenv`

### Models used
| Purpose | Model | Runs |
|---|---|---|
| Text embeddings | `all-MiniLM-L6-v2` | local, CPU |
| Multimodal embeddings | `clip-ViT-B-32` | local, CPU |
| Cross-encoder reranker | `cross-encoder/ms-marco-TinyBERT-L2-v2` | local, CPU |
| Query enhancement / rerank / RAG / vision | OpenRouter (`openrouter/free` >= `google/gemma-4-26b-a4b-it:free`) | remote API |

---

## Usage

Each CLI is independent of the others. A few representative commands:

```bash
# ---- Keyword search ----
python -m cli.keyword_search_cli bm25search "scary bear movie" --limit 5
python -m cli.keyword_search_cli bm25tf 42 "bear" 1.5 0.75

# ---- Semantic search ----
python -m cli.semantic_search_cli search_chunked "british bear with marmalade"
python -m cli.semantic_search_cli embed_chunks

# ---- Hybrid search ----
python -m cli.hybrid_search_cli weighted-search "action thriller" --alpha 0.5
python -m cli.hybrid_search_cli rrf-search "cute bear movie" --enhance rewrite
python -m cli.hybrid_search_cli rrf-search "scary movie" --rerank-method cross_encoder
python -m cli.hybrid_search_cli rrf-search "dinosaur park" --evaluate

# ---- RAG ----
python -m cli.augmented_generation_cli rag "It's my aunt's birthday and she loves British humor — what's a good movie?"
python -m cli.augmented_generation_cli summarize "marmalade bear movie"
python -m cli.augmented_generation_cli citations "movies about friendship between different species"
python -m cli.augmented_generation_cli question "What are the scariest bear movies?"

# ---- Evaluation ----
python -m cli.evaluation_cli --limit 5

# ---- Multimodal ----
python -m cli.multimodal_search_cli image_search data/paddington.jpeg
python -m cli.describe_image_cli --image data/paddington.jpeg --query "funny bear movie"
```

Run `python -m cli.<name>_cli --help` (or with no args) to see every available subcommand.

---

## Tests

```bash
uv run pytest            # or: python -m pytest
```

42 tests across `tests/test_{caches,search_utils,inverted_index,ranking,chunked_semantic_search,faiss_index,semantic_search,rerankers}.py`.

---

## What I Learned

- **Classic IR first**: building TF-IDF and BM25 by hand made the later neural pipeline far easier to reason about (why hybrid search matters, what RRF is fusing).
- **Embeddings + ANN**: cosine similarity, the embedding/query gap, and why FAISS (IndexFlatIP on normalized vectors) is exact cosine search.
- **Chunking is a precision lever**: chunk → re-score per document beat whole-document embedding on long texts.
- **Hybrid fusion beats either channel alone**: BM25 nails exact terms, semantics nail paraphrase — RRF needs no score normalization.
- **Two-stage retrieval (fetch 5×, then rerank)** is the standard way to reconcile recall and precision with expensive rerankers.
- **LLM reliability**: JSON output needs strict prompting; LLM rerankers need failure fallbacks (RRF order) and rate-limit delays.
- **Evaluation is non-negotiable**: a small golden dataset (P@k / R@k / F1) turns "feels better" into measurable progress.
- **Engineering hygiene pays off**: central config, result dataclasses, fingerprint-invalidated caches, and a lib/CLI split kept 54 commits of exploration maintainable.

---

## Roadmap / Next Steps

See `TODO.md`. Ideas already noted:
- Faster FAISS index families: **HNSW**, **IVF**, **LSH** (ANN speed vs. recall trade-offs).
- **ColBERT** / late chunking strategies.
- (Stretch) expand the golden dataset, add NDCG/other metrics, and serve via an API.