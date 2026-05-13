import argparse
import json
import string

from inverted_class import InvertedIndex
from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build index")
    args = parser.parse_args()

    translator = str.maketrans("", "", string.punctuation)

    # Handling file i/o
    with open("data/movies.json", "r", encoding="utf-8") as file:
        python_dict = json.load(file)

    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        content = file.read()

    stemmer = PorterStemmer()

    # if no movies in dict, then return empty
    movies = python_dict.get("movies", [])
    stopwords = content.splitlines()

    match args.command:
        case "search":
            # search query
            to_search = args.query
            search_token = [
                stemmer.stem(token)
                for token in to_search.translate(translator).lower().split()
                if token and token not in stopwords
            ]

            print(f"Searching for: {to_search}")
            results = []

            for movie in movies:
                # search target title
                title = movie.get("title", "")
                normalized_title = title.translate(translator).lower()

                if any(token in normalized_title for token in search_token):
                    results.append(title)

            if results:
                for index, result in enumerate(results, start=1):
                    print(f"{index}. {result}")
            else:
                print("Not found :( ")

        case "build":
            index_build = InvertedIndex()
            index_build.build()
            index_build.save()
            merida_docs = index_build.get_documents("merida")
            if merida_docs:
                print(f"First document for token 'merida' = {merida_docs[0]}")
            else:
                print("No documents for token 'merida'.")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
