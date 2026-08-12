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
        choices=["spell", "rewrite", "expand"],
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
            llm = LLMWrapper("google/gemma-4-26b-a4b-it:free")
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
            elif (args.enhance == "rewrite"):

                PROMPT = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{args.query}"
"""            
                response = llm.generate(PROMPT)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{response}'\n")
                args.query = response
            elif (args.enhance == "expand"):
                PROMPT = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

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
