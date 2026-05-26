import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)


def enhance_spelling(query: str) -> str:
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        config=types.GenerateContentConfig(
            system_instruction="""Fix any spelling errors in the user-provided movie search query below.
            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
            Preserve punctuation and capitalization unless a change is required for a typo fix.
            If there are no spelling errors, or if you're unsure, output the original query unchanged.
            Output only the final query text, nothing else.""",
            temperature=0.0,
            top_p=None,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        contents=f"User query: {query}",
    )
    if response and response.text:
        return response.text
    return query


def rewrite_query(query: str) -> str:
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        config=types.GenerateContentConfig(
            system_instruction="""Rewrite the user-provided movie search query below to be more specific and searchable.

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep the rewritten query concise (under 10 words)
            - It should be a Google-style search query, specific enough to yield relevant results
            - Don't use boolean logic

            Examples:
            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            If you cannot improve the query, output the original unchanged.
            Output only the rewritten query text, nothing else.""",
            temperature=0.1,
            top_p=0.10,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        contents=f"User query: {query}",
    )
    if response and response.text:
        return response.text
    return query


def query_expansion(query: str) -> str:
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        config=types.GenerateContentConfig(
            system_instruction="""Expand the user-provided movie search query below with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            Output only the additional terms; they will be appended to the original query.

            Examples:
            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted".
            """,
            temperature=0.6,
            top_p=0.8,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH, include_thoughts=False
            ),
        ),
        contents=f"User query: {query}",
    )
    if response and response.text:
        return response.text
    return query


def rerank_results_ind(query: str, title: str, desc: str):
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        config=types.GenerateContentConfig(
            system_instruction="""
            Rate how well this movie matches the search query.

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:
            """
        ),
        contents=f"""
        Query: "{query}"
        Movie: "{title} - {desc}"
        """,
    )
    if response and response.text:
        return int(response.text)

def re_rank_batch(query:str|None,doc:list[dict]|list|None):
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        config=types.GenerateContentConfig(
            system_instruction="""
            Rank the movies listed below by relevance to the following search query.
            Return the movie IDs in order of relevance, best match first.
            
            Your response must be a raw JSON array of integers.
            Do not wrap the JSON in Markdown. Do not use a ```json code block.
            Do not include any explanatory text.
            
            For example:
            [75, 12, 34, 2, 1]
            
            Ranking:"""
        ),
        contents=f"""
        Query: "{query}"
        Movies: "{doc}"
        """,
    )
    if response and response.text:
        return response.text

def re_rank_evaluate(query:str|None,doc:list[dict]|list|None):
    response = client.models.generate_content(
        model="gemma-4-31b-it",
        config=types.GenerateContentConfig(
            system_instruction="""
            Rate how relevant each result is to this query on a 0-3 scale:

            Scale:
            - 3: Highly relevant
            - 2: Relevant
            - 1: Marginally relevant
            - 0: Not relevant

            Do NOT give any numbers other than 0, 1, 2, or 3.

            Your response must be a raw JSON array of integers.
            Do not wrap the JSON in Markdown. Do not use a ```json code block.
            Do not include any explanatory text.

            For example:
            [2, 0, 3, 2, 0, 1]:"""
        ),
        contents=f"""
        Query: "{query}"
        Results:{chr(10).join(doc)}
        """,
    )
    if response and response.text:
        return response.text