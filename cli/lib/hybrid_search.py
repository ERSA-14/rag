import os

from .keyword_search import KeywordSearch
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.keyword_search = KeywordSearch()
        if not os.path.exists("cache/index.pkl"):
            self.keyword_search.build()
            self.keyword_search.save()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, str, float]]:
        self.keyword_search.load()
        return self.keyword_search.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
