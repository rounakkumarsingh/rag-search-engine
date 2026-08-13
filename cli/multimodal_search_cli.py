import argparse

from cli.lib.multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify a multimodal image embedding")
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)


if __name__ == "__main__":
    main()
