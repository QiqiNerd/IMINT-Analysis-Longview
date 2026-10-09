"""
Step 3: Analyze wrong classifications.

For every misclassified image, sends the image back to the model with a
diff-analysis meta-prompt and asks it to explain why the classification
was wrong. Writes diff_explanations.jsonl.
"""

import json
import time
from pathlib import Path

from openai import OpenAI


def load_eval_results(results_path: Path) -> list[dict]:
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_misclassifications(results: list[dict]) -> list[dict]:
    return [r for r in results if r["true_label"] != r["predicted_label"]]


def explain_error(
    client: OpenAI,
    model: str,
    meta_prompt_template: str,
    image_url: str,
    true_label: str,
    predicted_label: str,
    model_explanation: str,
) -> str:
    """Ask the model to explain why a classification was wrong."""
    prompt_text = meta_prompt_template.format(
        true_label=true_label,
        predicted_label=predicted_label,
        model_explanation=model_explanation,
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
                ],
            },
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def run_diff_analysis(
    client: OpenAI,
    model: str,
    eval_results_path: Path,
    meta_prompt_path: Path,
    output_dir: Path,
    sleep_sec: float = 0.2,
) -> list[dict]:
    """
    Analyze every misclassification and write diff_explanations.jsonl.

    Returns the list of diff entries.
    """
    meta_prompt_template = meta_prompt_path.read_text(encoding="utf-8")
    results = load_eval_results(eval_results_path)
    misclassified = get_misclassifications(results)

    if not misclassified:
        print("  No misclassifications found — nothing to analyze.")
        output_path = output_dir / "diff_explanations.jsonl"
        output_path.write_text("", encoding="utf-8")
        return []

    print(f"  {len(misclassified)} misclassifications to analyze")

    diff_entries = []
    for idx, item in enumerate(misclassified, 1):
        print(f"  [{idx}/{len(misclassified)}] {item['image_name']}", end=" ", flush=True)
        try:
            diff_explanation = explain_error(
                client=client,
                model=model,
                meta_prompt_template=meta_prompt_template,
                image_url=item["image_url"],
                true_label=item["true_label"],
                predicted_label=item["predicted_label"],
                model_explanation=item["explanation"],
            )
        except Exception as e:
            print(f"ERROR: {e}")
            diff_explanation = f"Analysis error: {e}"

        print("done")

        diff_entries.append({
            "image_name": item["image_name"],
            "image_url": item["image_url"],
            "true_label": item["true_label"],
            "predicted_label": item["predicted_label"],
            "model_explanation": item["explanation"],
            "diff_explanation": diff_explanation,
        })

        if idx < len(misclassified):
            time.sleep(sleep_sec)

    output_path = output_dir / "diff_explanations.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in diff_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(diff_entries)} diff explanations")
    return diff_entries
