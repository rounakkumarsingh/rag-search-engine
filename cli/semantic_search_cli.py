import argparse
import sys

from cli.lib.chunked_semantic_search import ChunkedSemanticSearch, semantic_chunks
from cli.lib.exceptions import EmptyQueryError
from cli.lib.movies import load_movies
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT
from cli.lib.semantic_search import SemanticSearch


def embed_text(text: str) -> None:
    ss = SemanticSearch(lambda: [])
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    ss = SemanticSearch(lambda: [])
    embedding = ss.generate_embedding(query)
    print(f"Text: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None:
    ss = SemanticSearch(load_movies)
    embeddings = ss.load_or_create_embeddings()
    print(f"Number of docs:   {len(ss.documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify model loading")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings cache against the corpus")

    query_embeddings_parser = subparsers.add_parser("embed_query", help="Embed a query")
    query_embeddings_parser.add_argument("text", type=str, help="Text to embed")

    search_parser = subparsers.add_parser("search", help="Search using whole-document embeddings")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Limit number of results")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Limit number of results")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text into semantic chunks")
    semantic_chunk_parser.add_argument("query", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="Max sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Sentence overlap between chunks")

    subparsers.add_parser("embed_chunks", help="Build chunked embeddings for all movies")

    args = parser.parse_args()

    match args.command:
        case "verify":
            ss = SemanticSearch(lambda: [])
            print(ss.verify())
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.text)
        case "search":
            ss = SemanticSearch(load_movies)
            ss.load_or_create_embeddings()
            try:
                results = ss.search(args.query, args.limit)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            for rank, (score, movie) in enumerate(results):
                print(f"{rank + 1}. {movie.get_title()} (score: {score})\n {movie.get_description()}")
        case "search_chunked":
            chunked_search = ChunkedSemanticSearch(load_movies)
            chunked_search.load_or_create_chunk_embeddings()
            try:
                results = chunked_search.search(args.query, args.limit)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            for i, (score, movie) in enumerate(results, start=1):
                print(f"\n{i}. {movie.get_title()} (score: {score:.4f})")
                print(f"   {movie.get_description()[:100]}...")
        case "semantic_chunk":
            chunks = semantic_chunks(args.query, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.query)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            chunked_search = ChunkedSemanticSearch(load_movies)
            embeddings = chunked_search.load_or_create_chunk_embeddings()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
