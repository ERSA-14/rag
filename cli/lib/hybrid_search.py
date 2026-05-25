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
        raw_scores = [score.get("score", 0.0) for score in bm_result]
        norm = normalize(raw_scores)
        for current_result, current_norm in zip(bm_result, norm):
            current_result["normalized_score"] = current_norm

        semantic_result = self.semantic_search.search_chunks(query, limit * 500)
        raw_scores = [score.get("score", 0.0) for score in semantic_result]
        norm = normalize(raw_scores)
        for current_result, current_norm in zip(semantic_result, norm):
            current_result["normalized_score"] = current_norm

        combined: dict[int, dict] = {}
        for result in bm_result:
            doc_id = result.get("id")
            if isinstance(doc_id, int):
                combined[doc_id] = {
                    "title": result.get("title"),
                    "description": result.get("description", ""),
                    "bm25_score": result.get("normalized_score"),
                    "semantic_score": 0.0,
                    "hybrid_score": 0.0,
                }
        for result in semantic_result:
            doc_id = result.get("id")
            if isinstance(doc_id, int):
                movie = self.keyword_search.docmap.get(doc_id)
                description_text = (
                    movie.get("description", "") if movie is not None else ""
                )
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
            hybrid_score = alpha * result.get("bm25_score", 0.0) + (
                1 - alpha
            ) * result.get("semantic_score", 0.0)
            result.update({"hybrid_score": hybrid_score})

        sorted_results = sorted(
            combined.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:limit]
        return sorted_results

    def rrf_search(self, query: str, k: int, limit: int = 10):
        bm_result = self.keyword_search.list_bm25_search(
            self._bm25_search(query, limit * 500)
        )
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)

        combined = {}
        for i, result in enumerate(bm_result, start=1):
            doc_id = result.get("id")
            if doc_id is not None:
                rrf = rrf_score(i, k)
                doc = self.keyword_search.docmap.get(doc_id, {})
                combined[doc_id] = {
                    "title": result["title"],
                    "document": doc,
                    "bm25_rank": i,
                    "semantic_rank": None,
                    "rrf_score": rrf,
                }
        for i, result in enumerate(semantic_result, start=1):
            doc_id = result.get("id")
            if doc_id is not None:
                if doc_id in combined:
                    if combined[doc_id]["semantic_rank"] is None:
                        existing_score = combined[doc_id]["rrf_score"]
                        combined[doc_id]["rrf_score"] = existing_score + rrf_score(i, k)
                        combined[doc_id]["semantic_rank"] = i
                else:
                    rrf = rrf_score(i, k)
                    doc = self.keyword_search.docmap.get(doc_id, {})
                    combined[doc_id] = {
                        "title": result["title"],
                        "document": doc,
                        "bm25_rank": None,
                        "semantic_rank": i,
                        "rrf_score": rrf,
                    }
        sorted_dict = sorted(
            combined.values(), key=lambda x: x["rrf_score"], reverse=True
        )[:limit]
        return sorted_dict


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


def rrf_score(rank: int, k: int = 60):
    return 1 / (k + rank)
