"""
Step 4: Cluster failure categories.

Reads diff_explanations.jsonl and sends all error explanations to the model
in a single call to identify common failure patterns. Writes clusters.json.
"""

import json
from pathlib import Path

from openai import OpenAI


def load_diff_explanations(diff_path: Path) -> list[dict]:
    entries = []
    with open(diff_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_diff_entries_for_prompt(entries: list[dict]) -> str:
    parts = []
    for i, e in enumerate(entries, 1):
        parts.append(
            f"Error {i}:\n"
            f"  Image: {e['image_name']}\n"
            f"  Predicted: {e['predicted_label']}, True: {e['true_label']}\n"
            f"  Explanation: {e['diff_explanation']}"
        )
    return "\n\n".join(parts)


def run_clustering(
    client: OpenAI,
    model: str,
    diff_explanations_path: Path,
    meta_prompt_path: Path,
    output_dir: Path,
) -> dict:
    """
    Cluster misclassification patterns and write clusters.json.

    Returns the parsed clusters dict.
    """
    entries = load_diff_explanations(diff_explanations_path)

    if not entries:
        print("  No diff explanations to cluster.")
        empty = {"categories": []}
        (output_dir / "clusters.json").write_text(
            json.dumps(empty, indent=2), encoding="utf-8"
        )
        return empty

    meta_template = meta_prompt_path.read_text(encoding="utf-8")
    diff_text = format_diff_entries_for_prompt(entries)

    prompt_text = meta_template.format(
        n_errors=len(entries),
        diff_entries=diff_text,
    )

    print(f"  Clustering {len(entries)} errors...")

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "failure_clusters",
                "schema": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "count": {"type": "integer"},
                                    "image_names": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["name", "description", "count", "image_names"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["categories"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        messages=[{"role": "user", "content": prompt_text}],
    )

    raw = (resp.choices[0].message.content or "").strip()
    clusters = json.loads(raw)

    output_path = output_dir / "clusters.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    for cat in clusters["categories"]:
        print(f"    - {cat['name']} ({cat['count']} errors): {cat['description'][:80]}")

    return clusters
