import argparse

from cli.lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify a multimodal image embedding")
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")

    image_search_parser = subparsers.add_parser("image_search", help="Search movies using an image")
    image_search_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)
    elif args.command == "image_search":
        results = image_search_command(args.image_path)
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result['title']} (similarity: {result['similarity']:.3f})")
            print(f"   {result['description']}")


if __name__ == "__main__":
    main()
