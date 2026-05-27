import argparse
import os

from huggingface_hub import logging as hf_logging
from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate an image embedding"
    )
    verify_parser.add_argument("image_path", type=str, help="Path to image")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movies using an image"
    )
    image_search_parser.add_argument(
        "image_path", type=str, help="Path to the query image"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
            hf_logging.set_verbosity_info()
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']} (similarity: {r['similarity']:.3f})")
                print(f"   {r['description']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
