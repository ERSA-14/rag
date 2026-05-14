import argparse, math

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

    tfidf_parser = subparsers.add_parser("tfidf", help="get term frequency and inverse document frequency")
    tfidf_parser.add_argument("document_id", type=int, help="Document ID")
    tfidf_parser.add_argument("word_to_search", type=str, help="Term to seach")

    args = parser.parse_args()

    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        content = file.read()

    stopwords = set(content.splitlines())

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

            search_token = index.tokens(to_search, stopwords)
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

            result = math.log((total_doc_count + 1)/ (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.wts}': {result:.2f}")

        case "tfidf":
            pass
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

            result = math.log((total_doc_count + 1)/ (term_match_doc_count + 1))

            # tf
            term_frequency = index.get_tf(id,term)

            tf_idf = result * term_frequency
            print(f"TF-IDF score of '{term}' in document '{id}': {tf_idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
