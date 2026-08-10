from cli.lib.movies import load_movies
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
    args = parser.parse_args()

    match args.command:
        case "verify":
            ss = SemanticSearch()
            ss.verify()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings(load_movies)
        case "embed_query":
            embed_query_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
