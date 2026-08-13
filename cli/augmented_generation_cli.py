import argparse
import sys

from cli.lib.config import LLM_DEFAULT_MODEL
from cli.lib.exceptions import EmptyQueryError
from cli.lib.hybrid_search import HybridSearch
from cli.lib.llm import LLMWrapper
from cli.lib.movies import load_movies
from cli.lib.ranking import RRF_K


def generate_answer(query: str, docs: str, llm: LLMWrapper) -> str:
    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""
    return llm.generate(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
            "rag", help="Perform RAG (search + generate answer)"
            )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to retrieve (default: 5)",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            search = HybridSearch(load_movies)
            try:
                results = search.rrf_search(query, RRF_K, args.top_k)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)

            docs = "\n".join(
                f"{idx}. {result.document.get_title()}: {result.document.get_description()}"
                for idx, result in enumerate(results, start=1)
            )

            llm = LLMWrapper(LLM_DEFAULT_MODEL)
            answer = generate_answer(query, docs, llm)

            print("Search Results:")
            for result in results:
                print(f"- {result.document.get_title()}")
            print("\nRAG Response:")
            print(answer)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
