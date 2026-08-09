import argparse
import sys

from cli.lib.inverted_index import InvertedIndex
from cli.lib.keyword_search import search_command
from cli.lib.movies import load_movies
from cli.lib.search_utils import tokenize_single_term

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index of documents")

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

    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_command(args.query)
            print(f"Searching for: {args.query}")
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.get_title()} (ID: {doc.get_id()})")
        case "build":
            ii = InvertedIndex(load_movies)
            ii.build()
            ii.save()
        case "tf":
            ii = InvertedIndex(load_movies)
            try:
                ii.load()
            except Exception:
                print("Index not found. Please build index first.")
                sys.exit(1)
            token = tokenize_single_term(args.term)
            print(ii.get_tf(args.doc_id, token))
        case "idf":
            ii = InvertedIndex(load_movies)
            try:
                ii.load()
            except Exception:
                print("Index not found. Please build index first.")
                sys.exit(1)
            token = tokenize_single_term(args.term)
            idf = ii.idf(token)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            ii = InvertedIndex(load_movies)
            try:
                ii.load()
            except Exception:
                print("Index not found. Please build index first.")
                sys.exit(1)
            token = tokenize_single_term(args.term)
            tf_idf = ii.tfidf(args.doc_id, token)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            ii = InvertedIndex(load_movies)
            try:
                ii.load()
            except Exception:
                print("Index not found. Please build index first.")
                sys.exit(1)
            token = tokenize_single_term(args.term)
            bm25idf = ii.get_bm25_idf(token)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
