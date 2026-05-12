import argparse
import json
import string
from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    translator = str.maketrans("","",string.punctuation)

    # Handling file i/o
    with open('data/movies.json',"r") as file:
        python_dict = json.load(file)
        file.close()

    with open("data/stopwords.txt","r") as file:
        content = file.read()
        file.close()

    stemmer = PorterStemmer()
        
	# if no movies in dict, then return empty
    movies = python_dict.get("movies",[])
    stopwords = content.splitlines()

    match args.command:
        case "search":
            # search query
            to_search = args.query
            search_token = [stemmer.stem(t) for t in to_search.translate(translator).lower().split(" ") if t and t not in stopwords]

            print(f"Searching for: {to_search}")
            results = []

            count = 0
            for movie in movies:
                # search target title 
                title = movie.get("title")
                token_title = [stemmer.stem(t) for t in title.translate(translator).lower().split(" ") if t and t not in stopwords]

                if any(var in var_title for var in search_token for var_title in token_title):
                    if count >= 5:
                        break
                    results.append(title)
                    count += 1

            if len(results) >= 0:
                i = 0
                for result in results:
                    i += 1
                    print(f"{i}. {result}")
            else:
                print("Not found :( ")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
