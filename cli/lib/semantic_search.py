import hashlib
import json
import os
import re
import numpy as np

from typing import Iterable
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv() 

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class SemanticSearch:
    def __init__(self):
        self.model = MODEL
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        string_representation = []

        for each_doc in self.documents:
            self.document_map[each_doc["id"]] = each_doc
            string_representation.append(
                f"{each_doc['title']}: {each_doc['description']}"
            )

        raw_embeddings = self.model.encode(
            string_representation, show_progress_bar=True
        )
        self.embeddings = raw_embeddings / np.linalg.norm(
            raw_embeddings, axis=1, keepdims=True
        )

        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for each_doc in documents:
            self.document_map[each_doc["id"]] = each_doc

        if os.path.exists("cache/movie_embeddings.npy"):
            loaded_embeddings = np.load("cache/movie_embeddings.npy")
            if len(loaded_embeddings) == len(self.documents):
                self.embeddings = loaded_embeddings / np.linalg.norm(
                    loaded_embeddings, axis=1, keepdims=True
                )
                return self.embeddings

        return self.build_embeddings(self.documents)

    def generate_embedding(self, text):
        stripped_text = text.strip()
        if not stripped_text:
            return np.array([])
        text_list = [stripped_text]
        embedding = self.model.encode(text_list, show_progress_bar=True)
        return embedding[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embed = self.generate_embedding(query)
        if query_embed.size == 0:
            return []

        query_norm = float(np.linalg.norm(query_embed))
        if query_norm == 0:
            return []
        norm_query = query_embed / query_norm

        scores = np.dot(self.embeddings, norm_query)
        top_indices = np.argsort(scores)[::-1][:limit]

        formatted_results = [
            {
                "score": float(scores[idx]),
                "title": self.documents[idx]["title"],
                "description": self.documents[idx]["description"],
            }
            for idx in top_indices
        ]
        return formatted_results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents

        list_dict: list[dict] = []
        all_chunk = []

        for movie_idx, each_doc in enumerate(self.documents):
            self.document_map[each_doc["id"]] = each_doc
            desc = each_doc["description"]

            if desc:
                chunk_list = semantic_chunk(desc, size=4, overlap=1)
                all_chunk.extend(chunk_list)
                if chunk_list:
                    for chunk_idx, _ in enumerate(chunk_list):
                        list_dict.append(
                            {
                                "movie_idx": movie_idx,
                                "chunk_idx": chunk_idx,
                                "total_chunks": len(chunk_list),
                            }
                        )
        if not all_chunk:
            self.chunk_embeddings = np.array([], dtype=np.float32).reshape(0, 0)
            self.chunk_metadata = []
        else:
            raw_chunks = self.model.encode(all_chunk, show_progress_bar=True)
            self.chunk_embeddings = raw_chunks / np.linalg.norm(
                raw_chunks, axis=1, keepdims=True
            )
            self.chunk_metadata = list_dict

        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        json_to_save = {
            "chunks": self.chunk_metadata,
            "total_chunks": len(all_chunk),
            "documents_fingerprint": get_documents_fingerprint(documents),
        }
        with open("cache/chunk_metadata.json", "w") as file:
            json.dump(json_to_save, file, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        for each_doc in documents:
            self.document_map[each_doc["id"]] = each_doc

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists(
            "cache/chunk_metadata.json"
        ):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as file:
                metadata = json.load(file)
                self.chunk_metadata = metadata["chunks"]
                cache_is_valid = (
                    len(self.chunk_embeddings) == metadata["total_chunks"]
                    and len(self.chunk_metadata) == metadata["total_chunks"]
                    and metadata.get("documents_fingerprint")
                    == get_documents_fingerprint(documents)
                )

                if cache_is_valid:
                    self.chunk_embeddings = self.chunk_embeddings / np.linalg.norm(
                        self.chunk_embeddings, axis=1, keepdims=True
                    )
                    return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):

        if self.chunk_embeddings is None or not self.documents or not self.chunk_metadata:
            return []

        if not isinstance(self.chunk_embeddings, Iterable):
            return []

        embed_query = self.generate_embedding(query)
        if embed_query is None or embed_query.size == 0:
            return []

        query_norm = float(np.linalg.norm(embed_query))
        if query_norm == 0:
            return []
        norm_query = embed_query / query_norm
        scores = np.dot(np.asarray(self.chunk_embeddings), norm_query)

        movies_score = {}
        for i, score in enumerate(scores):
            metadata_item = self.chunk_metadata[i]
            if metadata_item is not None:
                movie_idx = metadata_item.get("movie_idx")
                if movie_idx not in movies_score or score > movies_score[movie_idx].get(
                    "score"
                ):
                    movies_score[movie_idx] = {
                        "movie_idx": movie_idx,
                        "score": float(score),
                    }

        sorted_movies_score = sorted(
            movies_score.values(), key=lambda x: x.get("score"), reverse=True
        )[:limit]
        final_result: list[dict] = []

        for each_sorted_movies in sorted_movies_score:
            id = each_sorted_movies.get("movie_idx")
            doc = self.documents[id]
            final_result.append(
                {
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "document": doc.get("description", "")[:100],
                    "score": each_sorted_movies.get("score"),
                }
            )

        return final_result


def verify_model():
    semantic = SemanticSearch()
    print(f"Model loaded: {semantic}")
    print(f"Max sequence length: {semantic.model.max_seq_length}")


def verify_embeddings():
    semantic = SemanticSearch()
    with open("data/movies.json", "r") as file:
        documents = json.load(file)["movies"]
    embeddings = semantic.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
    print(f"Embeddings: {semantic.embeddings}")


def embed_text(text):
    semantic = SemanticSearch()

    output = semantic.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {output[:3]}")
    print(f"Dimensions: {output.shape[0]}")


def embed_query_text(query):
    semantic = SemanticSearch()

    embedding = semantic.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def hard_chunk(word: str, size: int, overlap: int):
    if overlap >= size:
        raise ValueError("Overlap must be smaller than size")
    if size < 1:
        raise ValueError("Size must be at least 1")
    new_word = word.split()
    chunks = []
    i, end, step = 0, len(new_word), size - overlap
    while i < end:
        chunk = " ".join(new_word[i : i + size])
        chunks.append(chunk)
        i += step
    return chunks


def semantic_chunk(word: str, size: int, overlap: int):
    if overlap >= size:
        raise ValueError("Overlap must be smaller than size")
    if size < 1:
        raise ValueError("Size must be at least 1")
    stripped_text = word.strip()
    if not stripped_text:
        return []
    word_list = re.split(r"(?<=[.!?])\s+", stripped_text)
    word_list = [sentence.strip() for sentence in word_list if sentence.strip()]
    if not word_list:
        return []
    chunks = []
    i = 0
    end = len(word_list)
    step = size - overlap
    while i < end:
        chunk = " ".join(word_list[i : i + size])
        if chunk:
            chunks.append(chunk.strip())
        if i + size >= end:
            break
        i += step
    return chunks


def get_documents_fingerprint(documents):
    text = json.dumps(documents, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
