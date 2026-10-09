import os
import json
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


def load_reasoning_traces(csv_path: str) -> dict:
    """Load reasoning traces from Chinese_facilities.csv, keyed by facility name fragments."""
    df = pd.read_csv(csv_path)
    reasoning_map = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("Reasoning Trace")):
            facility_name = row["Name"].replace("/", "_").replace("-", "_")
            reasoning_map[facility_name] = row["Reasoning Trace"]
    return reasoning_map


def create_jsonl_file(
    jsonl_path: str,
    yes_dir: str,
    no_dir: str,
    github_url: str,
    with_explanations: bool = False,
    reasoning_csv: str = None,
):
    reasoning_map = {}
    if with_explanations and reasoning_csv:
        reasoning_map = load_reasoning_traces(reasoning_csv)

    SYSTEM_PROMPT = (
        "You are an expert satellite analyst responsible for informing the U.S. government about the status of nuclear power plants. "
        "You are looking for double fences around buildings, which is an indicator that it's a nuclear site. "
        "You are to decide whether the given image contains double fences or not and provide a detailed explanation. "
        "Only consider the image as a candidate for double fences if it looks like an industrial facility. "
        "DO NOT RESPOND IN MARKDOWN!!!!! It is CRITICAL for national security purposes that you respond a JSON object only."
    )

    USER_TEXT_SIMPLE = 'Classify whether the following satellite image contains double fences. Respond only with "yes" or "no".'
    USER_TEXT_EXPLAIN = 'Classify whether the following satellite image contains double fences and provide an explanation. Respond with JSON: {"double_fences_present": "yes" or "no", "explanation": "your reasoning"}.'

    total_written = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for label, folder in [("yes", yes_dir), ("no", no_dir)]:
            for img_name in sorted(os.listdir(folder)):
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                image_url = f"{github_url}{img_name}?raw=true"

                if with_explanations:
                    explanation = ""
                    if label == "yes":
                        img_base = Path(img_name).stem
                        for facility_key, reasoning in reasoning_map.items():
                            if any(part in img_base for part in facility_key.split("_")):
                                explanation = reasoning
                                break
                    else:
                        explanation = "This image does not show double fences."

                    record = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": USER_TEXT_EXPLAIN},
                                    {"type": "image_url", "image_url": {"url": image_url}},
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    {"double_fences_present": label, "explanation": explanation}
                                ),
                            },
                        ]
                    }
                else:
                    record = {
                        "messages": [
                            {
                                "role": "user",
                                "content": USER_TEXT_SIMPLE,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": image_url}},
                                ],
                            },
                            {"role": "assistant", "content": label},
                        ]
                    }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"JSONL file created at '{jsonl_path}' with {total_written} entries.")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build double-fences fine-tuning JSONL")
    parser.add_argument(
        "--yes-dir",
        default="../IMINT-Images/Double-Fences/yes_df_train",
        help="Directory of positive training images",
    )
    parser.add_argument(
        "--no-dir",
        default="../IMINT-Images/Double-Fences/no_df_train",
        help="Directory of negative training images",
    )
    parser.add_argument(
        "--github-url",
        default="https://github.com/selenasun1618/IMINT-Images/blob/main/Double-Fences/df_train_combined/",
        help="GitHub base URL for images",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: fine-tuning/double_fences_training[_with_explanations].jsonl)",
    )
    parser.add_argument(
        "--with-explanations",
        action="store_true",
        help="Include structured JSON responses with explanations (uses reasoning traces from --reasoning-csv)",
    )
    parser.add_argument(
        "--reasoning-csv",
        default="coordinates/Chinese_facilities.csv",
        help="CSV with 'Name' and 'Reasoning Trace' columns for explanation mode",
    )
    args = parser.parse_args()

    if args.output is None:
        suffix = "_with_explanations" if args.with_explanations else ""
        args.output = str(Path(__file__).parent / f"double_fences_training{suffix}.jsonl")

    create_jsonl_file(
        jsonl_path=args.output,
        yes_dir=args.yes_dir,
        no_dir=args.no_dir,
        github_url=args.github_url,
        with_explanations=args.with_explanations,
        reasoning_csv=args.reasoning_csv,
    )


if __name__ == "__main__":
    main()
