import json, os, pickle, string
from typing import Any

from nltk.stem import PorterStemmer


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Any] = {}
        self.translator = str.maketrans("", "", string.punctuation)
        self.stemmer = PorterStemmer()

    def __add_document(self, doc_id: int, text: str) -> None:
        normalized = text.translate(self.translator).lower()
        for token in normalized.split():
            if token:
                stemmed = self.stemmer.stem(token)
                self.index.setdefault(stemmed, set()).add(doc_id)

    def get_document_id(self, search_target: str) -> list[int]:
        stemmed = self.stemmer.stem(search_target.lower())
        return sorted(list(self.index.get(stemmed, [])))

    def get_information(self, doc_id: int) -> Any:
        return self.docmap.get(doc_id)

    def build(self) -> None:
        with open("data/movies.json", "r", encoding="utf-8") as file:
            movie_dict = json.load(file)

        movies = movie_dict.get("movies", [])
        for movie in movies:
            doc_id = movie.get("id")
            movie_title = movie.get("title", "")
            movie_description = movie.get("description", "")

            if doc_id is None:
                continue

            self.__add_document(
                doc_id=doc_id, text=f"{movie_title} {movie_description}"
            )
            self.docmap.setdefault(doc_id, movie)

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)
            file.close()

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
            file.close()
            
    def load(self) -> None:
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Docmap file not found: {docmap_path}")

        with open(index_path, "rb") as file:
            self.index = pickle.load(file)
            file.close()

        with open(docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
            file.close()
