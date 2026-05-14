import argparse, string

from inverted_class import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build index")
    args = parser.parse_args()

    translator = str.maketrans("", "", string.punctuation)

    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        content = file.read()

    stopwords = content.splitlines()

    match args.command:
        case "search":
            # search query
            to_search = args.query
            normalized_query = to_search.translate(translator).lower()
            search_token = [
                token
                for token in normalized_query.split()
                if token and token not in stopwords
            ]

            print(f"Searching for: {to_search}")

            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Please run Build option first")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
