from cli.lib.llm import LLMWrapper
from cli.lib.movies import load_movies
from cli.lib.hybrid_search import HybridSearch, normalize
import argparse


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
        choices=["spell"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            hs = HybridSearch(load_movies)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for idx, (scores, doc) in enumerate(results, start=1):
                print(f"{idx}. {doc.get_title()}")
                print(f"  Hybrid Score: {scores['hybrid_score']:.3f}")
                print(f"  BM25 Rank: {scores['bm25_score'] or 0}, Semantic Rank: {scores['semantic_score'] or 0}")
                print(f"  {doc.get_description()[:100]}")
        case "rrf-search":
            hs = HybridSearch(load_movies)
            llm = LLMWrapper()
            if (args.enhance == "spell"):
                PROMPT = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{args.query}"
"""

                response = llm.generate(PROMPT)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{response}'\n")
                args.query = response

            results = hs.rrf_search(args.query, args.k, args.limit)
            for idx, (ranks, doc) in enumerate(results, start=1):
                print(f"{idx}. {doc.get_title()}")
                print(f"  Hybrid Score: {ranks['rrf_score']:.3f}")
                print(f"  BM25 Rank: {ranks['bm25_rank'] or 0}, Semantic Rank: {ranks['semantic_rank'] or 0}")
                print(f"  {doc.get_description()[:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
