import argparse
import json

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    hard_chunk,
    semantic_chunk,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify if the model is loaded")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Create embedding for a string"
    )
    embed_parser.add_argument("word", type=str, help="Word to embed")

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Create embedding for a string"
    )
    embed_query_parser.add_argument("word", type=str, help="query to embed")

    subparsers.add_parser(
        "verify_embeddings", help="verification for embedding creation"
    )

    embed_search_parser = subparsers.add_parser(
        "search", help="Search based on embedding"
    )
    embed_search_parser.add_argument("word", type=str, help="query to seacrh")
    embed_search_parser.add_argument(
        "--limit", type=int, default=5, help="limit the number of results (default: 5)"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="chunk texts for search/embedding"
    )
    chunk_parser.add_argument("word", type=str, help="query to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="chunk/bucket size"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="overlap words in chunk"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="semantic chunk texts for search/embedding"
    )
    semantic_chunk_parser.add_argument("word", type=str, help="query to semantic chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="chunk/bucket size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=1, help="overlap words in chunk"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="search in semantic chunks"
    )
    search_chunked_parser.add_argument("query", type=str, help="query to search for")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="number of results to return (default: 5)"
    )

    subparsers.add_parser(
        "embed_chunks", help="semantic chunk texts for search/embedding"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            word = args.word
            embed_text(word)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            word = args.word
            embed_query_text(word)

        case "search":
            limit = args.limit
            word = args.word

            semantic = SemanticSearch()

            with open("data/movies.json", "r") as file:
                documents = json.load(file)["movies"]
            semantic.load_or_create_embeddings(documents)

            results = semantic.search(word, limit)
            for index, result in enumerate(results, start=1):
                print(f"{index}. {result['title']} (score: {result['score']:.4f})")

        case "semantic_chunk":
            word = args.word
            size = args.max_chunk_size
            overlap = args.overlap
            chunks = semantic_chunk(word, size, overlap)

            print(f"Semantically chunking {len(word)} characters")
            for index, result in enumerate(chunks, start=1):
                print(f"{index}. {result}")

        case "embed_chunks":
            semantic = ChunkedSemanticSearch()

            with open("data/movies.json", "r") as file:
                documents = json.load(file)["movies"]
            embeddings = semantic.load_or_create_chunk_embeddings(documents)
            if embeddings is not None:
                print(f"Generated {len(embeddings)} chunked embeddings")

        case "chunk":
            word = args.word
            size = args.chunk_size
            overlap = args.overlap

            chunks = hard_chunk(word, size, overlap)
            print(f"Chunking {len(word)} characters")
            for index, result in enumerate(chunks, start=1):
                print(f"{index}. {result}")

        case "search_chunked":
            query = args.query
            limit = args.limit

            Csemantic = ChunkedSemanticSearch()
            with open("data/movies.json", "r") as file:
                documents = json.load(file)["movies"]
            embeddings = Csemantic.load_or_create_chunk_embeddings(documents)

            results = Csemantic.search_chunks(query, limit)
            for i, result in enumerate(results):
                print(
                    f"{i+1}. {result.get('title')} (score: {result.get('score'):.4f})"
                )
               
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
