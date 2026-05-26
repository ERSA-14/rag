import argparse
import json
import time

from lib.hybrid_search import HybridSearch, normalize
from lib.test_gemini import (
    enhance_spelling,
    query_expansion,
    re_rank_batch,
    re_rank_evaluate,
    rerank_results_ind,
    rewrite_query,
)
from sentence_transformers import CrossEncoder


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

    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="Reciprocal Rank Fusion weighted hybrid score based on the BM25 (keyword) and semantic scores (semantic)",
    )
    rrf_search_parser.add_argument(
        "query",
        type=str,
        help="search query",
    )
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="controls how much more weight we give to higher-ranked results vs. lower-ranked ones",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="how much result you want",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="re-rank method is provided, after doing the initial RRF search",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="use AI to evaluate results",
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
                print(f"{result.get('description', '')[:100]}")

        case "rrf-search":
            query = args.query
            k = args.k
            limit = args.limit

            with open("data/movies.json", "r") as file:
                documents = json.load(file)["movies"]
            hybrid = HybridSearch(documents)

            if args.enhance == "spell":
                query = enhance_spelling(query)

            if args.enhance == "rewrite":
                query = rewrite_query(query)

            if args.enhance == "expand":
                query = query_expansion(query)

            if args.rerank_method == "individual":
                results = hybrid.rrf_search(query, k, limit * 5)
                for result in results:
                    ai_score = rerank_results_ind(
                        query, result["title"], result["document"]["description"]
                    )
                    result["ai_score"] = ai_score
                    time.sleep(5)
                results.sort(key=lambda x: x["ai_score"], reverse=True)

            elif args.rerank_method == "batch":
                results = hybrid.rrf_search(query, k, limit * 5)
                new_results = [
                    {"document": doc["document"]}
                    for doc in results
                    if "document" in doc
                ]
                ai_response = re_rank_batch(query, new_results)
                if not ai_response:
                    print("Batch re-rank failed: empty response from model.")
                    return
                list_of_ai_results = json.loads(ai_response)
                for result in results:
                    id = result["document"]["id"]
                    result["ai_score"] = list_of_ai_results.index(id)

                results.sort(key=lambda x: x["ai_score"])

            elif args.rerank_method == "cross_encoder":
                results = hybrid.rrf_search(query, k, limit * 5)
                pairs = []
                for doc in results:
                    pairs.append(
                        [
                            query,
                            f"{doc.get('title', '')} - {doc.get('document', '')}",
                        ]
                    )
                cross_encoder = CrossEncoder(
                    "cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu"
                )
                scores = cross_encoder.predict(pairs)
                for result, score in zip(results, scores):
                    result["ai_score"] = float(score)
                results.sort(key=lambda x: x["ai_score"], reverse=True)
            else:
                results = hybrid.rrf_search(query, k, limit * 5)

            if args.evaluate:
                evaluation_items = [
                    f"{doc.get('title', '')} - {doc.get('document', {}).get('description', '')}"
                    for doc in results
                ]
                ai_response = re_rank_evaluate(query, evaluation_items)
                if not ai_response:
                    print("Evaluation failed: empty response from model.")
                    return
                scores = json.loads(ai_response)
                for result, score in zip(results, scores):
                    result["ai"] = score
                for i, result in enumerate(results[:limit]):
                    print(f"{i + 1}. {result['title']} {result.get('ai', 0)}/3")

            else:
                for i, result in enumerate(results[:limit]):
                    print(f"{i + 1}. {result['title']} ")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
