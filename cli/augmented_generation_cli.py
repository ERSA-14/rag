import argparse
from json import load

from lib.hybrid_search import HybridSearch
from lib.test_gemini import (
    augument_generation,
    citi_generation,
    qna_generation,
    sum_generation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--limit", type=int, default=5, help="limit Search query")

    sum_rag_parser = subparsers.add_parser("summarize", help="Summarize results")
    sum_rag_parser.add_argument("query", type=str, help="Search query for summaray")
    sum_rag_parser.add_argument(
        "--limit", type=int, default=5, help="limit Search query for summaray"
    )

    citi_rag_parser = subparsers.add_parser(
        "citations", help="Summarize results and sources"
    )
    citi_rag_parser.add_argument("query", type=str, help="Search query")
    citi_rag_parser.add_argument(
        "--limit", type=int, default=5, help="limit Search query"
    )

    question_rag_parser = subparsers.add_parser(
        "question", help="question and answer results and sources"
    )
    question_rag_parser.add_argument("query", type=str, help="Search query")
    question_rag_parser.add_argument(
        "--limit", type=int, default=5, help="limit Search query"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            with open("data/movies.json", "r") as file:
                documents = load(file)["movies"]
            results = HybridSearch(documents).rrf_search(args.query, 60, args.limit)
            pairs = [[doc.get("document", "")] for doc in results]
            ai_results = augument_generation(args.query, pairs)
            print(ai_results)

        case "summarize":
            with open("data/movies.json", "r") as file:
                documents = load(file)["movies"]
            results = HybridSearch(documents).rrf_search(args.query, 60, args.limit)
            pairs = [[doc.get("document", "")] for doc in results]
            ai_response = sum_generation(args.query, pairs)
            print(ai_response)

        case "citations":
            with open("data/movies.json", "r") as file:
                documents = load(file)["movies"]
            results = HybridSearch(documents).rrf_search(args.query, 60, args.limit)

            pairs = [[doc.get("document", "")] for doc in results]

            ai_response = citi_generation(args.query, pairs)
            for result in results:
                print(f"- {result['title']}")

            print()
            print(ai_response)

        case "question":
            with open("data/movies.json", "r") as file:
                documents = load(file)["movies"]
            results = HybridSearch(documents).rrf_search(args.query, 60, args.limit)
            pairs = [[doc.get("document", "")] for doc in results]
            ai_response = qna_generation(args.query, pairs)
            for result in results:
                print(f"- {result['title']}")

            print()
            print(ai_response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
