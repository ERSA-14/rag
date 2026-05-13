import json
import os
import pickle
import string
from typing import Any


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Any] = {}
        self.translator = str.maketrans("", "", string.punctuation)

    def __add_document(self, doc_id: int, text: str) -> None:
        normalized = text.translate(self.translator).lower()
        for token in normalized.split():
            if token:
                self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        return sorted(list(self.index.get(term.lower(), [])))

    def build(self) -> None:
        with open("data/movies.json", "r") as file:
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

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
