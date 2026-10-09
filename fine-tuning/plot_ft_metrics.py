import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def smooth(values, weight):
    """Apply exponential moving average smoothing.

    Args:
        values: Series or array of values to smooth
        weight: Smoothing weight between 0 and 1. Higher = smoother.
                0 = no smoothing, 0.99 = very smooth.
    """
    if weight == 0:
        return values
    smoothed = []
    last = values.iloc[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Plot fine-tuning metrics")
    parser.add_argument(
        "--smoothness", "-s",
        type=float,
        default=0.0,
        help="Smoothing weight (0-1). 0 = no smoothing, 0.9 = smooth, 0.99 = very smooth"
    )
    parser.add_argument("input_csv", type=Path, help="Fine-tuning metrics CSV")
    parser.add_argument("--output", type=Path, default=Path("fine_tuning_metrics.png"))
    args = parser.parse_args()

    if not 0 <= args.smoothness < 1:
        parser.error("Smoothness must be between 0 and 1 (exclusive)")

    df = pd.read_csv(args.input_csv)

    # Apply smoothing
    train_loss = smooth(df["train_loss"], args.smoothness)
    train_accuracy = smooth(df["train_mean_token_accuracy"], args.smoothness)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot training loss
    if args.smoothness > 0:
        ax1.plot(df["step"], df["train_loss"], color="tab:blue", alpha=0.2, label="raw")
        ax1.plot(df["step"], train_loss, color="tab:blue", alpha=0.9, label="smoothed")
        ax1.legend(loc="upper right")
    else:
        ax1.plot(df["step"], train_loss, color="tab:blue", alpha=0.7)
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Fine-tuning Metrics")
    ax1.grid(True, alpha=0.3)

    # Plot training accuracy
    if args.smoothness > 0:
        ax2.plot(df["step"], df["train_mean_token_accuracy"], color="tab:green", alpha=0.2, label="raw")
        ax2.plot(df["step"], train_accuracy, color="tab:green", alpha=0.9, label="smoothed")
        ax2.legend(loc="upper right")
    else:
        ax2.plot(df["step"], train_accuracy, color="tab:green", alpha=0.7)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Train Mean Token Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.show()

    print(f"Plotted {len(df)} steps (smoothness={args.smoothness})")
    print(f"Final loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final accuracy: {df['train_mean_token_accuracy'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
