# RAG CLI

A small CLI for keyword-based search and scoring over a movie dataset using an inverted index.

## Requirements

- Python >= 3.13
- `nltk==3.9.1`

## Data

The CLI expects the dataset at `data/movies.json` and stopwords at `data/stopwords.txt`.

## Usage

Build the index (creates cache files in `cache/`):

- `uv run cli/keyword_search_cli.py build`

Search by keyword(s) (returns up to five results):

- `uv run cli/keyword_search_cli.py search "your query here"`

Term frequency for a document:

- `uv run cli/keyword_search_cli.py tf <doc_id> <term>`

Inverse document frequency:

- `uv run cli/keyword_search_cli.py idf <term>`

TF-IDF score:

- `uv run cli/keyword_search_cli.py tfidf <doc_id> <term>`

BM25 IDF score:

- `uv run cli/keyword_search_cli.py bm25idf <term>`

BM25 TF score (optional `k1` and `b`):

- `uv run cli/keyword_search_cli.py bm25tf <doc_id> <term> [k1] [b]`

BM25 search (ranked results):

- `uv run cli/keyword_search_cli.py bm25search "your query here"`

## Output format

Search and BM25 search print the document id, title, and score (BM25 only).

## Notes

- Tokens are normalized by removing punctuation, lowercasing, removing stopwords, and stemming.
- Cache files are written under `cache/`.
- If you change tokenization or scoring logic, rebuild the index.
- `get_document_id` and `get_bm25_idf` expect a single-term input and raise if given multiple tokens.
- `bm25search` uses BM25 TF × IDF and ranks documents by descending score.
