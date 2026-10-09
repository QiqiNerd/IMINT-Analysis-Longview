#!/usr/bin/env python3
"""
Evaluate warhead detection on test set images.
Classifies images and compares predictions to ground truth labels.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Tuple
from openai import OpenAI
from dotenv import load_dotenv
import sys

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'scripts'))
sys.path.append(os.path.join(script_dir, '..', 'evals'))

from warhead_prompts import PROMPT as WARHEAD_PROMPT

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_label_from_filename(filename: str) -> str:
    """Extract ground truth label from filename.
    Positive examples start with location names (e.g., 'Ukrainka-1_...')
    Negative examples start with 'Negative_...'
    """
    if filename.startswith("Negative_"):
        return "no"
    else:
        return "yes"

def classify_image_local(client: OpenAI, model: str, image_path: str) -> Tuple[str, str]:
    """Classify a local image file and return (label, explanation)."""
    import base64

    # Read and encode image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        image_mime = "image/png" if image_path.lower().endswith('.png') else "image/jpeg"

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "warhead_presence",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "warhead_present": {"type": "string", "enum": ["yes", "no"]},
                            "explanation": {"type": "string"}
                        },
                        "required": ["warhead_present", "explanation"],
                        "additionalProperties": False
                    },
                    "strict": True,
                },
            },
            messages=[
                {
                    "role": "developer",
                    "content": WARHEAD_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Does this image contain a warhead storage facility?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_mime};base64,{image_data}"
                            }
                        },
                    ],
                },
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        ans = str(data.get("warhead_present", "no")).strip().lower()
        explanation = str(data.get("explanation", "")).strip()
        label = "yes" if ans == "yes" else "no"
        return (label, explanation)
    except Exception as e:
        print(f"Classification error for {image_path}: {e}")
        return ("no", f"Classification error: {str(e)}")

def run_test_eval(test_dir: str, output_csv: str, model: str = "gpt-4o-2024-08-06", sleep: float = 0.2):
    """Run evaluation on test set images."""
    import time

    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Get all image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        image_files.extend(test_path.glob(f'*{ext}'))

    image_files = sorted(image_files)
    total = len(image_files)

    print(f"Found {total} images in test set")
    print(f"Model: {model}")
    print(f"Output: {output_csv}")
    print()

    results = []

    for idx, img_path in enumerate(image_files, 1):
        # Extract ground truth
        ground_truth = extract_label_from_filename(img_path.name)

        # Classify
        print(f"[{idx}/{total}] Classifying: {img_path.name}")
        predicted_label, explanation = classify_image_local(client, model, str(img_path))

        # Determine if correct
        is_correct = (predicted_label == ground_truth)

        results.append({
            "image_name": img_path.name,
            "ground_truth": ground_truth,
            "predicted": predicted_label,
            "correct": is_correct,
            "explanation": explanation
        })

        status = "✓" if is_correct else "✗"
        print(f"  {status} Ground truth: {ground_truth}, Predicted: {predicted_label}")

        if idx < total:
            time.sleep(sleep)

    # Write results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "ground_truth", "predicted", "correct", "explanation"])
        writer.writeheader()
        writer.writerows(results)

    # Calculate metrics
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0

    true_positives = sum(1 for r in results if r["ground_truth"] == "yes" and r["predicted"] == "yes")
    false_positives = sum(1 for r in results if r["ground_truth"] == "no" and r["predicted"] == "yes")
    false_negatives = sum(1 for r in results if r["ground_truth"] == "yes" and r["predicted"] == "no")
    true_negatives = sum(1 for r in results if r["ground_truth"] == "no" and r["predicted"] == "no")

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total images: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives:  {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  True Negatives:  {true_negatives}")
    print()
    print("Metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1_score:.3f}")
    print()
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate warhead detection on test set")
    parser.add_argument("--test-dir",
                       default="../IMINT-Images/Russian_Warhead_Storage/warheads_test_combined",
                       help="Directory containing test images")
    parser.add_argument("--output",
                       default="eval_results/warhead_test_eval.csv",
                       help="Output CSV file path")
    parser.add_argument("--model",
                       default="gpt-4o-2024-08-06",
                       help="OpenAI model to use")
    parser.add_argument("--sleep",
                       type=float,
                       default=0.2,
                       help="Sleep seconds between API calls")

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_test_eval(args.test_dir, args.output, args.model, args.sleep)
