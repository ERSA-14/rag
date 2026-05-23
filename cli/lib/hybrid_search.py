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
        self.keyword_search.load()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, str, float]]:
        self.keyword_search.load()
        return self.keyword_search.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm_result = self.keyword_search.list_bm25_search(
            self._bm25_search(query, limit * 500)
        )
        raw_scores = [score.get("score") for score in bm_result]
        norm = normalize(raw_scores)
        for current_result, current_norm in zip(bm_result, norm):
            current_result["normalized_score"] = current_norm

        semantic_result = self.semantic_search.search_chunks(query, limit * 500)
        raw_scores = [score.get("score") for score in semantic_result]
        norm = normalize(raw_scores)
        for current_result, current_norm in zip(semantic_result, norm):
            current_result["normalized_score"] = current_norm

        combined = {}
        for result in bm_result:
            doc_id = result.get("id")
            movie = self.keyword_search.docmap[doc_id]
            combined[doc_id] = {
                "title": result.get("title"),
                "description": movie.get("description"),
                "bm25_score": result.get("normalized_score"),
                "semantic_score": 0.0,
                "hybrid_score": 0.0,
            }
        for result in semantic_result:
            doc_id = result.get("id")
            movie = self.keyword_search.docmap.get(doc_id, {})
            description_text = movie.get("description")
            if doc_id in combined:
                if result["normalized_score"] > combined[doc_id]["semantic_score"]:
                    combined[doc_id]["semantic_score"] = result["normalized_score"]
            else:
                combined[doc_id] = {
                    "title": result.get("title"),
                    "description": description_text,
                    "bm25_score": 0.0,
                    "semantic_score": result.get("normalized_score"),
                    "hybrid_score": 0.0,
                }
        for result in combined.values():
            hybrid_score = alpha * result.get("bm25_score") + (1 - alpha) * result.get(
                "semantic_score"
            )
            result.update({"hybrid_score": hybrid_score})

        sorted_results = sorted(
            combined.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:limit]
        return sorted_results

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(score_list: list[float | int]):
    if not score_list:
        return []
    min_val = min(score_list)
    max_val = max(score_list)
    if max_val == min_val:
        return [1.0] * len(score_list)

    result = []
    for val in score_list:
        if val == min_val:
            result.append(0.0)
        elif val == max_val:
            result.append(1.0)
        else:
            result.append((val - min_val) / (max_val - min_val))
    return result
