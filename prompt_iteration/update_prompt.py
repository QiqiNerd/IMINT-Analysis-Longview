"""
Step 5: Generate an updated prompt using failure clusters.

Sends the current prompt + cluster analysis to the model and asks it to
rewrite the prompt to address the identified failure patterns.
Writes updated_prompt.txt.
"""

import json
from pathlib import Path

from openai import OpenAI


def format_clusters_for_prompt(clusters: dict) -> str:
    parts = []
    for cat in clusters.get("categories", []):
        parts.append(
            f"- {cat['name']} ({cat['count']} errors): {cat['description']}"
        )
    return "\n".join(parts) if parts else "(no failure categories identified)"


def run_prompt_update(
    client: OpenAI,
    model: str,
    current_prompt: str,
    clusters_path: Path,
    metrics_path: Path,
    meta_prompt_path: Path,
    output_dir: Path,
) -> str:
    """
    Generate an updated prompt and write it to updated_prompt.txt.

    Returns the updated prompt text.
    """
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta_template = meta_prompt_path.read_text(encoding="utf-8")

    clusters_text = format_clusters_for_prompt(clusters)

    prompt_text = meta_template.format(
        current_prompt=current_prompt,
        clusters=clusters_text,
        accuracy=metrics["accuracy"],
        n_correct=metrics["correct"],
        n_total=metrics["total"],
    )

    print("  Generating updated prompt...")

    resp = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt_text}],
    )

    updated_prompt = (resp.choices[0].message.content or "").strip()

    output_path = output_dir / "updated_prompt.txt"
    output_path.write_text(updated_prompt, encoding="utf-8")

    print(f"  Updated prompt written to: {output_path}")
    return updated_prompt
