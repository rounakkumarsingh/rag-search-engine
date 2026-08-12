import argparse
import sys

from cli.lib.exceptions import EmptyQueryError
from cli.lib.inverted_index import InvertedIndex
from cli.lib.keyword_search import bm25_search_command, bm25_tf_command, search_command
from cli.lib.movies import load_movies
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B, tokenize_single_term


def load_index() -> InvertedIndex:
    try:
        return InvertedIndex.load_or_build(load_movies)
    except Exception as exc:
        print(f"Failed to load or build index: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to look up")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to look up")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "search":
            try:
                results = search_command(args.query)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"Searching for: {args.query}")
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.get_title()} (ID: {doc.get_id()})")
        case "build":
            ii = InvertedIndex(load_movies)
            ii.build()
            ii.save()
        case "tf":
            token = tokenize_single_term(args.term)
            print(load_index().get_tf(args.doc_id, token))
        case "idf":
            token = tokenize_single_term(args.term)
            idf = load_index().idf(token)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            token = tokenize_single_term(args.term)
            tf_idf = load_index().tfidf(args.doc_id, token)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            token = tokenize_single_term(args.term)
            bm25idf = load_index().get_bm25_idf(token)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(str(args.doc_id), args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            try:
                results = bm25_search_command(args.query, args.limit)
            except EmptyQueryError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            for i, (score, doc) in enumerate(results, 1):
                print(f"{i}. ({doc.get_id()}) {doc.get_title()} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
