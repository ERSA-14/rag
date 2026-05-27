# RAG CLI

A CLI toolkit for movie search using keyword (BM25), semantic (all-MiniLM-L6-v2), hybrid, and multimodal (CLIP) retrieval, plus Gemini-based query enhancement, re-ranking, and augmented generation.

## Requirements

- Python >= 3.13
- Dependencies listed in `pyproject.toml`

## Installation

```bash
uv sync
```

## Data

### `data/movies.json` (5000 movies)

```json
{
  "movies": [
    {
      "id": 1,
      "title": "dhurandhar",
      "description": "A badly injured ..."
    },
    ...
  ]
}
```

### `data/test_dataset.json` 

```json
[
  {
    "query": "indian films",
    "relevant_docs": ["dhurandhar, ..."]
  },
  ...
]
```

### `data/stopwords.txt`

One stopword per line (198 words).


### Cache files (`cache/`)

| File | Description |
|---|---|
| `index.pkl` | Inverted index (keyword search) |
| `docmap.pkl` | Document ID to document mapping |
| `term_frequencies.pkl` | Term frequency per document |
| `doc_lengths.pkl` | Document lengths |
| `movie_embeddings.npy` | Pre-computed semantic embeddings |
| `chunk_embeddings.npy` | Pre-computed chunk embeddings |
| `chunk_metadata.json` | Chunk metadata |

## Library Modules

### `cli/lib/keyword_search.py` — `KeywordSearch`

| Method | Input | Output |
|---|---|---|
| `tokens(text: str)` | Raw text | `list[str]` — stemmed, stopword-filtered tokens |
| `get_document_id(search_target: str)` | Single term | `list[int]` — matching document IDs |
| `get_information(doc_id: int)` | Document ID | Movie dict (`{id, title, description}`) |
| `build()` | — | Builds inverted index from `data/movies.json` |
| `save()` | — | Pickles cache files to `cache/` |
| `load()` | — | Loads cache files from `cache/` |
| `get_tf(doc_id: int, term: str)` | Doc ID + term | `int` — raw term frequency |
| `get_word_count(doc_id)` | Doc ID | `int` — total words in document |
| `total_items()` | — | `int` — total indexed documents |
| `get_bm25_idf(search_target: str)` | Single term | `float` — BM25 IDF score |
| `get_bm25_tf(doc_id, term, k1=1.5, b=0.75)` | Doc ID + term + params | `float` — BM25 TF score |
| `bm25(doc_id, term)` | Doc ID + term | `float` — BM25 score (IDF × TF) |
| `bm25_search(query, limit)` | Query string + limit | `list[(doc_id, title, score)]` |
| `list_bm25_search(query_list)` | BM25 result tuples | `list[{id, title, description, score}]` |

### `cli/lib/semantic_search.py` — `SemanticSearch` & `ChunkedSemanticSearch`

Model: `all-MiniLM-L6-v2`

| Method | Input | Output |
|---|---|---|
| `build_embeddings(documents)` | `list[dict]` movie list | `None` (saves to `cache/movie_embeddings.npy`) |
| `load_or_create_embeddings(documents)` | `list[dict]` movie list | `np.ndarray` — loaded or built embeddings |
| `generate_embedding(text)` | String | `np.ndarray` — single embedding vector |
| `search(query, limit=5)` | Query string + limit | `list[{score, title, description}]` |
| `build_chunk_embeddings(documents)` | `list[dict]` movie list | `None` (saves to `cache/`) |
| `load_or_create_chunk_embeddings(documents)` | `list[dict]` movie list | `np.ndarray` — loaded or built chunk embeddings |
| `search_chunks(query, limit=10)` | Query string + limit | `list[{id, title, description, score}]` |

**Helper functions:**

| Function | Input | Output |
|---|---|---|
| `verify_model()` | — | Prints model info |
| `verify_embeddings()` | — | Prints embeddings shape |
| `embed_text(text)` | String | Prints first 3 dims + shape |
| `embed_query_text(query)` | Query string | Prints first 3 dims + shape |
| `cosine_similarity(vec1, vec2)` | Two vectors | `float` |
| `hard_chunk(word, size, overlap)` | Text + params | `list[str]` — fixed-size word chunks |
| `semantic_chunk(word, size, overlap)` | Text + params | `list[str]` — sentence-bounded chunks |
| `get_documents_fingerprint(documents)` | `list[dict]` | SHA-256 hex string |

### `cli/lib/hybrid_search.py` — `HybridSearch`

| Method | Input | Output |
|---|---|---|
| `__init__(documents: list[dict])` | Movie list | Loads/builds keyword + chunked semantic indexes |
| `weighted_search(query, alpha=0.5, limit=5)` | Query + alpha weight + limit | `list[{id, title, description, hybrid_score}]` — BM25 × α + semantic × (1-α) |
| `rrf_search(query, k=60, limit=10)` | Query + RRF constant + limit | `list[{id, title, description, rrf_score}]` — reciprocal rank fusion |

**Helper functions:**

| Function | Input | Output |
|---|---|---|
| `normalize(score_list)` | List of scores | `list[float]` — min-max normalized to [0, 1] |
| `rrf_score(rank, k=60)` | Rank + k | `float` — `1 / (k + rank)` |

### `cli/lib/multimodal_search.py` — `MultimodalSearch`

Model: `clip-ViT-B-32`

| Method | Input | Output |
|---|---|---|
| `__init__(documents, model_name, device)` | Movie list (optional), model name, device hint | Encodes all document texts |
| `embed_image(image_path: str)` | Path to image | `np.ndarray` — image embedding vector |
| `search_with_image(image_path: str)` | Path to image | `list[{id, title, description, similarity}]` — top 5 by cosine similarity |

**Helper functions:**

| Function | Input | Output |
|---|---|---|
| `verify_image_embedding(image_path: str)` | Image path | Prints embedding dimension |
| `image_search_command(image_path: str)` | Image path | Returns top 5 results (used by CLI) |

### `cli/lib/test_gemini.py` — Gemini-based AI functions (model: `gemma-4-31b-it`)

| Function | Input | Output |
|---|---|---|
| `enhance_spelling(query: str)` | Query string | `str` — spelling-corrected query |
| `rewrite_query(query: str)` | Query string | `str` — rewritten (< 10 words) |
| `query_expansion(query: str)` | Query string | `str` — expanded with synonyms |
| `rerank_results_ind(query, title, desc)` | Query + title + desc | `int` — relevance score 0-10 |
| `re_rank_batch(query, docs)` | Query + doc list | `str` — JSON array of ranked IDs |
| `re_rank_evaluate(query, docs)` | Query + doc list | `str` — JSON array of 0-3 ratings |
| `augument_generation(query, docs)` | Query + doc list | `str` — natural language answer |
| `sum_generation(query, docs)` | Query + doc list | `str` — 3-4 sentence summary |
| `citi_generation(query, docs)` | Query + doc list | `str` — answer with citations |
| `qna_generation(query, docs)` | Query + doc list | `str` — conversational Q&A |

## CLI Commands

Run all commands from the project root via `uv run cli/<module>.py <command> [args]`.

### `cli/keyword_search_cli.py`

| Command | Arguments | Description |
|---|---|---|
| `build` | — | Build inverted index from `data/movies.json` |
| `search` | `<query>` | Print matching doc IDs and titles (top 5) |
| `tf` | `<doc_id> <term>` | Raw term frequency |
| `idf` | `<term>` | Inverse document frequency |
| `tfidf` | `<doc_id> <term>` | TF-IDF score |
| `bm25idf` | `<term>` | BM25 IDF score |
| `bm25tf` | `<doc_id> <term> [k1] [b]` | BM25 TF score |
| `bm25search` | `<query>` | BM25 ranked search (top 5) |

### `cli/semantic_search_cli.py`

| Command | Arguments | Description |
|---|---|---|
| `verify` | — | Print model info |
| `embed_text` | `<text>` | Embed and print first 3 dims |
| `embed_query` | `<text>` | Embed query and print shape |
| `verify_embeddings` | — | Verify embedding creation |
| `search` | `<query> [--limit]` | Semantic similarity search |
| `chunk` | `<text> [--chunk-size] [--overlap]` | Hard-chunk text |
| `semantic_chunk` | `<text> [--max-chunk-size] [--overlap]` | Sentence-bounded chunking |
| `search_chunked` | `<query> [--limit]` | Search over semantic chunks |
| `embed_chunks` | — | Build and save chunk embeddings |

### `cli/hybrid_search_cli.py`

| Command | Arguments | Description |
|---|---|---|
| `normalize` | `<scores...>` | Min-max normalize scores |
| `weighted-search` | `<query> [--alpha] [--limit]` | Weighted BM25 + semantic search |
| `rrf-search` | `<query> [-k] [--limit] [--enhance] [--rerank-method] [--evaluate]` | Reciprocal rank fusion with optional query enhancement and AI re-ranking |

### `cli/multimodal_search_cli.py`

| Command | Arguments | Description |
|---|---|---|
| `verify_image_embedding` | `<image_path>` | Generate and print image embedding info |
| `image_search` | `<image_path>` | Search movies by image (CLIP-based, top 5) |

### `cli/augmented_generation_cli.py`

| Command | Arguments | Description |
|---|---|---|
| `rag` | `<query> [--limit]` | RAG answer using retrieved docs |
| `summarize` | `<query> [--limit]` | 3-4 sentence summary |
| `citations` | `<query> [--limit]` | Answer with citations |
| `question` | `<query> [--limit]` | Conversational Q&A |

### `cli/describe_image_cli.py`

```bash
uv run cli/describe_image_cli.py --image <path> --query "<text>"
```

Rewrites a search query based on image content using Gemma 4-31B-it.

### `cli/evaluation_cli.py`

```bash
uv run cli/evaluation_cli.py [--limit 5]
```

Evaluates Precision@k, Recall@k, and F1 on `data/test_dataset.json` using hybrid RRF search.

## Environment

Set `GEMINI_API_KEY` in a `.env` file at the project root (used by `test_gemini.py`).

## Notes

- Keyword search tokenization: strip punctuation → lowercase → remove stopwords → Porter stem.
- Semantic search uses L2-normalized embeddings; cosine similarity via dot product.
- All cache files live under `cache/`. Delete them to force a rebuild.
- For `bm25tf`, default `k1=1.5`, default `b=0.75`.
