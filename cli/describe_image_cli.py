import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Query Rewrite CLI")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image",
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Search query to rewrite"
    )

    args = parser.parse_args()

    client = genai.Client(api_key=api_key)

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    query = args.query.strip()
    image_path = args.image
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as file:
        image_content = file.read()

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        config=types.GenerateContentConfig(system_instruction=system_prompt),
        contents=[
            types.Part.from_bytes(data=image_content, mime_type=mime),
            f"User query: {query}",
        ],
    )
    if response and response.text:
        print(f"Rewritten query: {response.text.strip()}")
    else:
        print("No response text returned.")
    if response and response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
