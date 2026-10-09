#!/usr/bin/env python3
"""Build an OpenAI fine-tuning JSONL file for the AAA classification task.

This utility prepares records only. Uploading the file and starting a fine-tuning
job should be done separately with credentials supplied through the environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PROMPT = (
    "Classify whether the following satellite image contains any anti-aircraft "
    'artillery (AAA). Respond only with "yes" or "no".'
)


def iter_images(directory: Path):
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def image_url(base_url: str, image_path: Path) -> str:
    return f"{base_url.rstrip('/')}/{image_path.name}"


def make_record(url: str, label: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
            {"role": "assistant", "content": label},
        ]
    }


def create_jsonl_file(
    positive_dir: Path,
    negative_dir: Path,
    output_path: Path,
    image_base_url: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with output_path.open("w", encoding="utf-8") as output:
        for directory, label in ((positive_dir, "yes"), (negative_dir, "no")):
            if not directory.is_dir():
                raise FileNotFoundError(f"Image directory not found: {directory}")
            for image_path in iter_images(directory):
                record = make_record(image_url(image_base_url, image_path), label)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-base-url",
        required=True,
        help="Public or authorized base URL containing the referenced images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = create_jsonl_file(
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        output_path=args.output,
        image_base_url=args.image_base_url,
    )
    print(f"Wrote {total} training records to {args.output}")


if __name__ == "__main__":
    main()
