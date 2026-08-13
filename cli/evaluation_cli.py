from cli.lib.ranking import RRF_K
from cli.lib.hybrid_search import HybridSearch
from cli.lib.movies import load_movies
from cli.lib.chunked_semantic_search import ChunkedSemanticSearch
from cli.lib.config import PROJECT_ROOT
from cli.lib.caches import load_json
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    args = parser.parse_args()
    limit = args.limit

    golden_dataset = load_json(PROJECT_ROOT / "data" / "golden_dataset.json")
    hs = HybridSearch(load_movies)
    print(f"k = {limit}")
    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        results = hs.rrf_search(query, RRF_K, limit)
        cnt = 0
        correct_retrieved = []
        for result in results:
            if result.document.get_title() in test_case["relevant_docs"]:
                correct_retrieved.append(result.document.get_title())
                cnt += 1
        precision = cnt / len(results)
        recall =  cnt / test_case["relevant_docs"]
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"Retrieved: {", ".join(correct_retrieved)}")
        print(f"Relevant: {", ".join(test_case["relevant_docs"])}")


    # run evaluation logic here

if __name__ == "__main__":
    main()
