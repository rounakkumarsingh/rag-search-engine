from cli.lib.config import LLM_RERANK_MODEL
from cli.lib.llm import LLMWrapper
from cli.lib.movies import load_movies
from cli.lib.hybrid_search import HybridSearch
from cli.lib.prompts import expand_query_prompt, rewrite_query_prompt, spell_fix_prompt
from cli.lib.ranking import normalize
from cli.lib.rerankers import make_reranker
import argparse

ENHANCE_PROMPTS = {
    "spell": spell_fix_prompt,
    "rewrite": rewrite_query_prompt,
    "expand": expand_query_prompt,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor (default: 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Reciprocal rank fusion hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="Fusion constant (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")
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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            hs = HybridSearch(load_movies)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for idx, result in enumerate(results, start=1):
                print(f"{idx}. {result.document.get_title()}")
                print(f"  Hybrid Score: {result.hybrid_score:.3f}")
                print(f"  BM25 Score: {result.bm25_score:.3f}, Semantic Score: {result.semantic_score:.3f}")
                print(f"  {result.document.get_description()[:100]}")
        case "rrf-search":
            hs = HybridSearch(load_movies)
            llm = LLMWrapper(LLM_RERANK_MODEL)
            query = args.query
            if args.enhance:
                prompt = ENHANCE_PROMPTS[args.enhance](query)
                response = llm.generate(prompt)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{response}'\n")
                query = response

            rr_limit = args.limit * 5 if args.rerank_method != "none" else args.limit
            results = hs.rrf_search(query, args.k, rr_limit)

            reranker = make_reranker(args.rerank_method, llm)
            if reranker is not None:
                print(f"Re-ranking top {len(results)} results using {args.rerank_method} method...")
                results = reranker.rerank(query, results)[:args.limit]

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
