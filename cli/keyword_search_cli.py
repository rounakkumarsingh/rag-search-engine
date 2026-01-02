import argparse

from cli.lib.build import build
from cli.lib.idf import IDFService
from cli.lib.search import SearchService
from cli.lib.tf import TFService

SEARCH_SERVICE = SearchService()
TF_SERVICE = TFService()
IDF_SERVICE = IDFService()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="search query")
    subparsers.add_parser("build", help="Build up the index and save it")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument(
        "doc_id",
        type=int,
        help="The document's identifier on which qury will be performed",
    )
    tf_parser.add_argument("term", type=str, help="The term whose frequency is needed")
    idf_parser = subparsers.add_parser("idf", help="Get term's idf")
    idf_parser.add_argument("term", type=str, help="The term whose idf is needed")
    tf_idf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tf_idf_parser.add_argument(
        "doc_id",
        type=int,
        help="The document's identifier on which qury will be performed",
    )
    tf_idf_parser.add_argument("term", type=str, help="The term whose TF-IDF is needed")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            matches = SEARCH_SERVICE.search(args.query)
            for i, match in enumerate(matches):
                print(f"{i + 1}. {match.title}")
        case "build":
            build()
        case "tf":
            print(TF_SERVICE.find_tf(args.doc_id, args.term))
        case "idf":
            idf = IDF_SERVICE.idf_score(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = TF_SERVICE.find_tf(args.doc_id, args.term) * IDF_SERVICE.idf_score(
                args.term
            )
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
