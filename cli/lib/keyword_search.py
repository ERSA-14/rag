import json
import math
import os
import pickle
import string
from collections import Counter
from typing import Any

from constants import BM25_B, BM25_K1
from nltk.stem import PorterStemmer


class KeywordSearch:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Any] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.stopwords: set[str] = set()

        self.translator = str.maketrans("", "", string.punctuation)
        self.stemmer = PorterStemmer()

        self.__load_stopwords()

    def tokens(self, text: str) -> list[str]:
        tokens = text.translate(self.translator).lower().split()
        tokens = filter(lambda token: token not in self.stopwords, tokens)
        tokens = map(self.stemmer.stem, tokens)
        return list(tokens)

    def __load_stopwords(self) -> None:
        stopword_file_path = "data/stopwords.txt"
        if not os.path.exists(stopword_file_path):
            self.stopwords = set()
            raise FileNotFoundError(f"Stopfile not found: {stopword_file_path}")
        else:
            with open(stopword_file_path, "r", encoding="utf-8") as file:
                content = file.read()
            self.stopwords = set(content.splitlines())

    def __add_document(self, doc_id: int, text: str) -> None:
        stemmed_tokens = self.tokens(text)

        length = len(stemmed_tokens)
        self.doc_lengths.setdefault(doc_id, length)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        self.term_frequencies[doc_id].update(stemmed_tokens)

        for token in stemmed_tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_document_id(self, search_target: str) -> list[int]:
        tokens = self.tokens(search_target)
        if len(tokens) != 1:
            raise ValueError("Expected a single token term.")
        stemmed = tokens[0]
        return sorted(list(self.index.get(stemmed, [])))

    def get_information(self, doc_id: int) -> Any:
        return self.docmap.get(doc_id)

    def build(self) -> None:
        if not os.path.exists("data/movies.json"):
            print("Movies data file missing.")
            return
        with open("data/movies.json", "r", encoding="utf-8") as file:
            movie_dict = json.load(file)

        movies = movie_dict.get("movies", [])
        for movie in movies:
            doc_id = movie.get("id")
            movie_title = movie.get("title", "")
            movie_description = movie.get("description", "")

            if doc_id is None:
                continue

            self.__add_document(doc_id, f"{movie_title} {movie_description}")
            self.docmap.setdefault(doc_id, movie)

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open("cache/doc_lengths.pkl", "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self) -> None:
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        term_freq_path = "cache/term_frequencies.pkl"
        doc_lengths_path = "cache/doc_lengths.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Docmap file not found: {docmap_path}")
        if not os.path.exists(term_freq_path):
            raise FileNotFoundError(f"Frequency file not found: {term_freq_path}")
        if not os.path.exists(doc_lengths_path):
            raise FileNotFoundError(f"Frequency file not found: {doc_lengths_path}")

        with open(index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        with open(term_freq_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open(doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = self.tokens(term)
        stemmed = tokens[0] if tokens else None
        if not stemmed:
            return 0
        return self.term_frequencies.get(doc_id, Counter()).get(stemmed, 0)

    def get_word_count(self, doc_id) -> int:
        return sum(self.term_frequencies.get(doc_id, {}).values())

    def total_items(self) -> int:
        return len(self.docmap)

    def get_bm25_idf(self, search_target: str) -> float:
        tokens = self.tokens(search_target)
        stemmed = tokens[0] if tokens else None
        if not stemmed:
            return 0.0
        N = self.total_items()
        df = len(self.index.get(stemmed, ()))

        IDF = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        return IDF

    def bm25_search(self, query: str, limit: int) -> list[tuple[int, str, float]]:
        tokens = self.tokens(query)
        if not tokens or limit <= 0:
            return []

        idf_by_token = {token: self.get_bm25_idf(token) for token in tokens}
        scores: dict[int, float] = {}
        candidate_docs = set()
        for token in tokens:
            candidate_docs.update(self.index.get(token, set()))

        for doc_id in candidate_docs:
            score = 0.0
            for token in tokens:
                score += self.get_bm25_tf(doc_id, token) * idf_by_token[token]
            scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results: list[tuple[int, str, float]] = []
        for doc_id, score in ranked[:limit]:
            title = self.docmap.get(doc_id, {}).get("title", "")
            results.append((doc_id, title, score))

        return results

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        avg_len = self.__get_avg_doc_length()
        doc_len = self.doc_lengths.get(doc_id, 0)
        length_norm = 1 - b + b * (doc_len / avg_len if avg_len else 0.0)
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm) if tf else 0.0

    def __get_avg_doc_length(self) -> float:
        total_docs = len(self.doc_lengths)
        total_word = sum(self.doc_lengths.values())
        return total_word / total_docs if total_docs > 0 else 0.0

    def bm25(self, doc_id, term) -> float:
        return self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)
