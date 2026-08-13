import argparse
import json
import logging
import sys

from cli.lib.config import (
    DEFAULT_SEARCH_LIMIT,
    LLM_RERANK_MODEL,
    RERANK_FETCH_MULTIPLIER,
)
from cli.lib.exceptions import EmptyQueryError
from cli.lib.llm import LLMWrapper
from cli.lib.movies import load_movies
from cli.lib.hybrid_search import HybridSearch
from cli.lib.prompts import expand_query_prompt, rewrite_query_prompt, spell_fix_prompt
from cli.lib.ranking import RRF_K, normalize
from cli.lib.rerankers import make_reranker

logger = logging.getLogger(__name__)

ENHANCE_PROMPTS = {
    "spell": spell_fix_prompt,
    "rewrite": rewrite_query_prompt,
    "expand": expand_query_prompt,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor (default: 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results (default: 5)")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Reciprocal rank fusion hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, default=RRF_K, help="Fusion constant (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results (default: 5)")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        default="individual",
        choices=["individual", "batch", "cross_encoder", "none"],
        help="Result reranking method (default: individual)",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Use an LLM to rate result relevance on a 0-3 scale",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        for name in ("cli", "__main__"):
            logging.getLogger(name).setLevel(logging.DEBUG)

    match args.command:
        case "normalize":
            for score in normalize(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            hs = HybridSearch(load_movies)
            try:
                results = hs.weighted_search(args.query, args.alpha, args.limit)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result.document.get_title()}")
                print(f"  Hybrid Score: {result.hybrid_score:.3f}")
                print(f"  BM25 Score: {result.bm25_score:.3f}, Semantic Score: {result.semantic_score:.3f}")
                print(f"  {result.document.get_description()[:100]}")
        case "rrf-search":
            hs = HybridSearch(load_movies)
            llm = LLMWrapper(LLM_RERANK_MODEL)
            query = args.query
            logger.debug("Original query: %s", query)
            if args.enhance:
                prompt = ENHANCE_PROMPTS[args.enhance](query)
                response = llm.generate(prompt)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{response}'\n")
                query = response
            logger.debug("Query after enhancement (%s): %s", args.enhance, query)

            try:
                rr_limit = args.limit * RERANK_FETCH_MULTIPLIER if args.rerank_method != "none" else args.limit
                results = hs.rrf_search(query, args.k, rr_limit)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            logger.debug("RRF search returned %d results", len(results))
            for idx, result in enumerate(results, start=1):
                logger.debug(
                    "  %d. %s (rrf_score=%.3f, bm25_rank=%s, semantic_rank=%s)",
                    idx,
                    result.document.get_title(),
                    result.rrf_score,
                    result.bm25_rank,
                    result.semantic_rank,
                )

            reranker = make_reranker(args.rerank_method, llm)
            if reranker is not None:
                print(f"Re-ranking top {len(results)} results using {args.rerank_method} method...")
                results = reranker.rerank(query, results)[:args.limit]
                logger.debug("Re-ranking completed; final %d results", len(results))
                for idx, result in enumerate(results, start=1):
                    logger.debug(
                        "  %d. %s (rr_score=%s, rr_rank=%s, rrf_score=%.3f)",
                        idx,
                        result.document.get_title(),
                        result.rr_score,
                        result.rr_rank,
                        result.rrf_score,
                    )

            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result.document.get_title()}")
                if args.rerank_method == "individual":
                    print(f"  Re-rank Score: {result.rr_score:.3f}/10")
                elif args.rerank_method == "batch":
                    print(f"  Re-rank Rank: {result.rr_rank}")
                elif args.rerank_method == "cross_encoder":
                    print(f"  Re-rank Score: {result.rr_score:.3f}")
                print(f"  RRF Score: {result.rrf_score:.3f}")
                print(f"  BM25 Rank: {result.bm25_rank or 0}, Semantic Rank: {result.semantic_rank or 0}")
                print(f"  {result.document.get_description()[:100]}")

            if args.evaluate:
                print()
                evaluate_results(query, results, llm)
        case _:
            parser.print_help()


def evaluate_results(query: str, results: list, llm: LLMWrapper) -> None:
    formatted_results = "\n".join(
        f"{idx}. {result.document.get_title()}: {result.document.get_description()}"
        for idx, result in enumerate(results, start=1)
    )
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{formatted_results}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    try:
        response = llm.generate(prompt)
        scores = json.loads(response.strip())
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return

    for idx, result in enumerate(results, start=1):
        score = scores[idx - 1] if idx - 1 < len(scores) else "?"
        print(f"{idx}. {result.document.get_title()}: {score}/3")


if __name__ == "__main__":
    main()
