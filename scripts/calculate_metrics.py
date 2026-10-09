#!/usr/bin/env python3
"""
Calculate precision, recall, and F1 score from CSV with human_label and model_label columns.
"""

import pandas as pd
import argparse
from pathlib import Path

def calculate_metrics(csv_file):
    """Calculate precision, recall, and F1 score from CSV file."""

    # Read CSV
    df = pd.read_csv(csv_file)

    # Validate required columns
    if 'human_label' not in df.columns or 'model_label' not in df.columns:
        raise ValueError("CSV must contain 'human_label' and 'model_label' columns")

    # Convert to binary (assuming 'yes'/'no' labels)
    human_binary = (df['human_label'] == 'yes').astype(int)
    model_binary = (df['model_label'] == 'yes').astype(int)

    # Calculate confusion matrix components
    true_positives = ((human_binary == 1) & (model_binary == 1)).sum()
    false_positives = ((human_binary == 0) & (model_binary == 1)).sum()
    false_negatives = ((human_binary == 1) & (model_binary == 0)).sum()
    true_negatives = ((human_binary == 0) & (model_binary == 0)).sum()

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(df)

    # Print results
    print(f"📊 **Evaluation Metrics for {Path(csv_file).name}**")
    print(f"Total samples: {len(df)}")
    print()
    print(f"**Confusion Matrix:**")
    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"True Negatives:  {true_negatives}")
    print()
    print(f"**Metrics:**")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1_score:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate precision, recall, and F1 score from CSV")
    parser.add_argument("csv_file", help="Path to CSV file with human_label and model_label columns")

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.csv_file).exists():
        print(f"❌ Error: File '{args.csv_file}' not found")
        return

    try:
        calculate_metrics(args.csv_file)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
