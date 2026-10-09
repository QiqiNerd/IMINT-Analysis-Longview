"""
Step 2: Run classification eval on a dataset split.

Sends each image to the model with the given prompt, collects structured
predictions, and writes eval_results.jsonl + metrics.json.
"""

import json
import time
from pathlib import Path

from openai import OpenAI


def load_dataset(jsonl_path: str) -> list[dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line)["item"])
    return items


def classify_image(
    client: OpenAI,
    model: str,
    developer_prompt: str,
    user_prompt: str,
    image_url: str,
    response_schema: dict,
    label_field: str,
) -> tuple[str, str]:
    """Send one image to the model and return (predicted_label, explanation)."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_schema", "json_schema": response_schema},
        messages=[
            {"role": "developer", "content": developer_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
                ],
            },
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = json.loads(raw)
    label = str(data.get(label_field, "no")).strip().lower()
    explanation = str(data.get("explanation", "")).strip()
    return label, explanation


def run_eval(
    client: OpenAI,
    task_config: dict,
    model: str,
    prompt_text: str,
    jsonl_path: str,
    output_dir: Path,
    sleep_sec: float = 0.2,
) -> dict:
    """
    Evaluate the prompt on every item in the dataset.

    Returns a metrics dict and writes:
      - output_dir/eval_results.jsonl
      - output_dir/metrics.json
    """
    label_field = task_config["label_field"]
    items = load_dataset(jsonl_path)
    total = len(items)

    results = []
    for idx, item in enumerate(items, 1):
        image_url = item["image_url"]
        true_label = item[label_field]

        print(f"  [{idx}/{total}] {item['image_name']}", end=" ", flush=True)
        try:
            predicted, explanation = classify_image(
                client=client,
                model=model,
                developer_prompt=task_config["developer_prompt"],
                user_prompt=prompt_text,
                image_url=image_url,
                response_schema=task_config["response_schema"],
                label_field=label_field,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            predicted, explanation = "no", f"API error: {e}"

        correct = predicted == true_label
        print("OK" if correct else "WRONG")

        results.append({
            "image_name": item["image_name"],
            "image_url": image_url,
            "true_label": true_label,
            "predicted_label": predicted,
            "explanation": explanation,
        })

        if idx < total:
            time.sleep(sleep_sec)

    # Write results
    results_path = output_dir / "eval_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute metrics
    n_correct = sum(1 for r in results if r["true_label"] == r["predicted_label"])
    tp = sum(1 for r in results if r["true_label"] == "yes" and r["predicted_label"] == "yes")
    fp = sum(1 for r in results if r["true_label"] == "no" and r["predicted_label"] == "yes")
    fn = sum(1 for r in results if r["true_label"] == "yes" and r["predicted_label"] == "no")
    tn = sum(1 for r in results if r["true_label"] == "no" and r["predicted_label"] == "no")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "total": total,
        "correct": n_correct,
        "accuracy": n_correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Accuracy: {metrics['accuracy']:.1%} ({n_correct}/{total})")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    return metrics
