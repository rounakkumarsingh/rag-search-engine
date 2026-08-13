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


def summarize(query: str, results: str, llm: LLMWrapper) -> str:
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    return llm.generate(prompt)


def answer_with_citations(query: str, documents: str, llm: LLMWrapper) -> str:
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

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

    summarize_parser = subparsers.add_parser(
            "summarize", help="Summarize search results for a query"
            )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summary"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize (default: 5)",
    )

    citations_parser = subparsers.add_parser(
            "citations", help="Answer a query with citations from search results"
            )
    citations_parser.add_argument(
        "query", type=str, help="Search query for the answer"
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to cite (default: 5)",
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
        case "summarize":
            query = args.query

            search = HybridSearch(load_movies)
            try:
                results = search.rrf_search(query, RRF_K, args.limit)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)

            results_str = "\n".join(
                f"{idx}. {result.document.get_title()}: {result.document.get_description()}"
                for idx, result in enumerate(results, start=1)
            )

            llm = LLMWrapper(LLM_DEFAULT_MODEL)
            summary = summarize(query, results_str, llm)

            print("Search Results:")
            for result in results:
                print(f"- {result.document.get_title()}")
            print("\nLLM Summary:")
            print(summary)
        case "citations":
            query = args.query

            search = HybridSearch(load_movies)
            try:
                results = search.rrf_search(query, RRF_K, args.limit)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)

            documents = "\n".join(
                f"{idx}. {result.document.get_title()}: {result.document.get_description()}"
                for idx, result in enumerate(results, start=1)
            )

            llm = LLMWrapper(LLM_DEFAULT_MODEL)
            answer = answer_with_citations(query, documents, llm)

            print("Search Results:")
            for result in results:
                print(f"- {result.document.get_title()}")
            print("\nLLM Answer:")
            print(answer)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
