# RAG CLI

A small CLI for keyword-based search over a movie dataset using an inverted index.

## Requirements

- Python >= 3.13
- `nltk==3.9.1`

## Data

The CLI expects the dataset at `data/movies.json` and stopwords at `data/stopwords.txt`.

## Usage

Build the index (creates `cache/index.pkl` and `cache/docmap.pkl`):

- `uv run cli/keyword_search_cli.py build`

Search by keyword(s) (returns up to five results):

- `uv run cli/keyword_search_cli.py search "your query here"`

## Output format

Search prints the document id and title for each match, for example:

- `123: Some Movie Title`

## Notes

- Tokens are normalized by removing punctuation and lowercasing.
- Stopwords are filtered using `data/stopwords.txt`.
- The index stores stemmed tokens (Porter stemmer).
