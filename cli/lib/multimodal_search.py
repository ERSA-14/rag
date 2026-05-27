import json

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(
        self,
        documents: list[dict] | None = None,
        model_name: str = "clip-ViT-B-32",
        device: str | None = None,
    ) -> None:
        resolved_device = None
        if device:
            if device == "gpu":
                device = "cuda"
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            resolved_device = device

        if resolved_device:
            self.model = SentenceTransformer(model_name, device=resolved_device)
        else:
            self.model = SentenceTransformer(model_name)

        self.documents = documents or []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        if self.texts:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        else:
            self.text_embeddings = np.array([])

    def embed_image(self, image_path: str):
        with Image.open(image_path) as image:
            embedding = self.model.encode([image], show_progress_bar=True)[0]
        return embedding

    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        scores = []
        for i, text_emb in enumerate(self.text_embeddings):
            sim = np.dot(image_embedding, text_emb) / (
                np.linalg.norm(image_embedding) * np.linalg.norm(text_emb)
            )
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, sim in scores[:5]:
            doc = self.documents[i]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity": float(sim),
                }
            )

        return results


def verify_image_embedding(image_path: str) -> None:
    multi_model = MultimodalSearch()
    embedding = multi_model.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    with open("data/movies.json") as f:
        data = json.load(f)

    search = MultimodalSearch(documents=data["movies"])
    return search.search_with_image(image_path)
