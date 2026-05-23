import argparse
import json

from lib.hybrid_search import HybridSearch, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="List of scores to normalize (e.g., 0.1 0.5 0.9)",
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="weighted hybrid score based on the BM25 (keyword) and semantic scores (semantic)",
    )
    weighted_search_parser.add_argument(
        "query",
        type=str,
        help="search query",
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="control how much keyword vs semantic in query",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="how much result you want",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            score_list = args.scores
            scores = normalize(score_list)
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            query = args.query
            alpha = args.alpha
            limit = args.limit
            with open("data/movies.json", "r") as file:
                documents = json.load(file)["movies"]
            hybrid = HybridSearch(documents)
            results = hybrid.weighted_search(query, alpha, limit)
            for i, result in enumerate(results):
                print(f"{i + 1})  {result.get('title')}")
                print(f"Hybrid Score: {result.get('hybrid_score')}")
                print(
                    f"BM25: {result.get('bm25_score')}, Semantic: {result.get('semantic_score')}"
                )
                print(f"{result.get('description')[:100]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
