import argparse
import json

from lib.hybrid_search import HybridSearch


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

    with open("data/movies.json", "r") as file:
        documents = json.load(file)["movies"]
    hybrid = HybridSearch(documents)

    with open("data/test_dataset.json", "r") as file:
        tests = json.load(file)["test_cases"]

    print(f"k={limit}\n")
    for test in tests:
        query = test.get("query", "")
        relevant_docs = test.get("relevant_docs", [])
        relevant_set = set(relevant_docs)
        relevant_count = len(relevant_set)

        rrf_results = hybrid.rrf_search(query, 60, limit)
        retrieved = [result["document"]["title"] for result in rrf_results]
        retrieved_count = len(retrieved)
        relevant_retrieved = sum(1 for title in retrieved if title in relevant_set)
        precision = relevant_retrieved / retrieved_count if retrieved_count else 0.0
        recall_relevant = sum(1 for title in relevant_set if title in retrieved)
        recall = recall_relevant / relevant_count if relevant_count else 0.0
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}\n")


if __name__ == "__main__":
    main()
