import json
import os
import pickle
import string
from collections import Counter
from typing import Any
from nltk.stem import PorterStemmer


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Any] = {}
        self.translator = str.maketrans("", "", string.punctuation)
        self.stemmer = PorterStemmer()
        self.term_frequencies: dict[int, Counter[str]] = {}

    def _normalize(self, text: str) -> str:
        return text.translate(self.translator).lower()

    def _token_iter(self, text: str):
        return filter(None, self._normalize(text).split())

    def tokens(
        self, text: str, stopwords: set[str] | None = None, stem: bool = False
    ) -> list[str]:
        tokens = self._token_iter(text)

        if stopwords:
            tokens = filter(lambda token: token not in stopwords, tokens)

        if stem:
            tokens = map(self.stemmer.stem, tokens)
        return list(tokens)

    def stem_term(self, term: str) -> str | None:
        return next(iter(self.tokens(term, stem=True)), None)

    def __add_document(self, doc_id: int, text: str) -> None:
        stemmed_tokens = self.tokens(text, stem=True)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(stemmed_tokens)

        for token in stemmed_tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_document_id(self, search_target: str) -> list[int]:
        stemmed = self.stem_term(search_target)
        if not stemmed:
            return []
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

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self) -> None:
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        term_freq_path = "cache/term_frequencies.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Docmap file not found: {docmap_path}")
        if not os.path.exists(term_freq_path):
            raise FileNotFoundError(f"Docmap file not found: {term_freq_path}")

        with open(index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        with open(term_freq_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

    def get_tf(self, doc_id: int, term: str) -> int:
        stemmed = self.stem_term(term)
        if not stemmed:
            return 0
        return self.term_frequencies.get(doc_id, Counter()).get(stemmed, 0)

    def get_word_count(self,doc_id) -> int:
        return sum(self.term_frequencies.get(doc_id,{}).values())

    def total_items(self) -> int:
        return len(self.docmap)
