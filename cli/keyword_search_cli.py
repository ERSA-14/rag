import argparse
import math

from constants import BM25_B, BM25_K1
from inverted_class import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build index")

    tf_parser = subparsers.add_parser("tf", help="get Term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("wts", type=str, help="Get Inverse word frequency count")
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="get term frequency and inverse document frequency"
    )
    tfidf_parser.add_argument("document_id", type=int, help="Document ID")
    tfidf_parser.add_argument("word_to_search", type=str, help="Term to seach")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # search query
            to_search = args.query

            print(f"Searching for: {to_search}")

            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Please build Index first")

            search_token = index.tokens(to_search)
            if not search_token:
                print("Not found :( ")
                return

            results: list[int] = []
            seen: set[int] = set()

            for token in search_token:
                doc_ids = index.get_document_id(token)
                for doc_id in doc_ids:
                    if doc_id in seen:
                        continue
                    results.append(doc_id)
                    seen.add(doc_id)
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break

            if results:
                for doc_id in results:
                    doc = index.get_information(doc_id) or {}
                    title = doc.get("title", "")
                    print(f"{doc_id}: {title}")
            else:
                print("Not found :( ")

        case "build":
            index_build = InvertedIndex()
            index_build.build()
            index_build.save()

        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Please build Index first")

            term_frequency = index.get_tf(args.doc_id, args.term)
            print(str(term_frequency))

        case "idf":
            word = args.wts
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Please build Index first")

            term_match_doc_count = len(index.get_document_id(word))
            total_doc_count = index.total_items()

            result = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.wts}': {result:.2f}")

        case "tfidf":
            id = args.document_id
            term = args.word_to_search

            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Please build Index first")

            # idf
            term_match_doc_count = len(index.get_document_id(term))
            total_doc_count = index.total_items()

            result = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            # tf
            term_frequency = index.get_tf(id, term)
            term_frequency = index.get_tf(id, term)

            tf_idf = result * term_frequency
            print(f"TF-IDF score of '{term}' in document '{id}': {tf_idf:.2f}")

        case "bm25idf":
            word = args.term
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Please build Index first")

            result = index.get_bm25_idf(word)
            print(f"BM25 IDF score of '{word}': {result:.2f}")

        case "bm25tf":
            word = args.term
            doc_id = args.doc_id
            k1 = args.k1
            b = args.b

            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Please build Index first")

            result = index.get_bm25_tf(doc_id, word, k1, b)
            print(f"BM25 TF score of '{word}' in document '{doc_id}': {result:.2f}")

        case "bm25search":
            word = args.query
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Please build Index first")

            for doc_id, title, score in index.bm25_search(word, limit=5):
                print(f"({doc_id}) {title} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
