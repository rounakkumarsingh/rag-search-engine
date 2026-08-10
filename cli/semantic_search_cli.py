import re
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT
from cli.lib.movies import MovieDocument, load_movies
from cli.lib.semantic_search import SemanticSearch, embed_text, verify_embeddings, embed_query_text
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model loading")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify model loading")
    query_embeddings_parser = subparsers.add_parser("embed_query", help="Verify model loading")
    query_embeddings_parser.add_argument("text", type=str, help="Text to embed")

    search_parser = subparsers.add_parser("search", help="Verify model loading")
    search_parser.add_argument("query", type=str, help="Text to embed")
    search_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Limit number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Verify model loading")
    chunk_parser.add_argument("query", type=str, help="Text to embed")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="Chunk size")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Chunk size")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Verify model loading")
    semantic_chunk_parser.add_argument("query", type=str, help="Text to embed")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=4, help="Chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Chunk size")

    args = parser.parse_args()

    match args.command:
        case "verify":
            ss = SemanticSearch(lambda: [])
            ss.verify()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings(load_movies)
        case "embed_query":
            embed_query_text(args.text)
        case "search":
            query = args.query
            ss = SemanticSearch(load_movies)
            ss.load_or_create_embeddings()
            result = ss.search(query, args.limit)
            for rank, (score, doc) in enumerate(result):
                movie = doc
                print(f"{rank + 1}. {movie.get_title()} (score: {score})\n {movie.get_description()}")
        case "chunk":
            query = args.query
            n = args.chunk_size
            words = query.split()
            print(f"Chunking {len(query)} characters")
            cnt = 0
            while cnt < len(words):
                print(f"{(cnt + n)//n}. {" ".join(words[max(0, cnt - args.overlap): cnt + n])}")
                cnt += n
        case "semantic_chunk":
            sentences = re.split(r"(?<=[.!?])\s+", args.query)
            n = args.max_chunk_size
            print(f"Semantically chunking {len(args.query)} characters")
            cnt = 0
            while cnt < len(sentences):
                print(f"{(cnt + n)//n}. {" ".join(sentences[max(0, cnt - args.overlap): max(0,cnt - args.overlap) + n])}")
                # print(f"{max(0, cnt - args.overlap), cnt + n}")
                cnt = max(0, cnt-args.overlap) + n
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
