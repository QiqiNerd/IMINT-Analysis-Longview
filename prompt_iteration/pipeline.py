#!/usr/bin/env python3
"""
Prompt iteration pipeline for satellite image classification.

Prompts are versioned at the run level as prompt_v0.txt (seed), prompt_v1.txt,
prompt_v2.txt, etc. The pipeline auto-detects the latest version, or you can
override with --prompt-version N.

Usage:
    # First iteration on validation set (uses seed prompt, saved as prompt_v0)
    python pipeline.py iterate --task aaa --seed prompts/aaa_seed.txt

    # Continue iterating (auto-detects latest prompt version)
    python pipeline.py iterate --task aaa --run-dir runs/aaa_20260320_143000

    # Continue but use a specific prompt version
    python pipeline.py iterate --task aaa --run-dir runs/aaa_20260320_143000 --prompt-version 1

    # Run final evaluation on test set (auto-detects latest prompt)
    python pipeline.py test --task aaa --run-dir runs/aaa_20260320_143000

    # Test with a specific prompt version
    python pipeline.py test --task aaa --run-dir runs/aaa_20260320_143000 --prompt-version 2
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import TASKS, MODEL, PROMPTS_DIR, RUNS_DIR
from evaluate import run_eval
from analyze_diffs import run_diff_analysis
from cluster import run_clustering
from update_prompt import run_prompt_update


def init_client() -> OpenAI:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def create_run_dir(task: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{task}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_latest_iteration(run_dir: Path) -> int:
    """Find the highest-numbered iteration_N directory."""
    max_iter = 0
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try:
                n = int(d.name.split("_")[1])
                max_iter = max(max_iter, n)
            except (IndexError, ValueError):
                pass
    return max_iter


# ── Prompt versioning ────────────────────────────────────────────────

def get_prompt_version_path(run_dir: Path, version: int) -> Path:
    return run_dir / f"prompt_v{version}.txt"


def get_latest_prompt_version(run_dir: Path) -> int:
    """Return the highest prompt version number found, or -1 if none."""
    max_v = -1
    for f in run_dir.iterdir():
        if f.is_file() and f.name.startswith("prompt_v") and f.name.endswith(".txt"):
            try:
                v = int(f.stem.split("_v")[1])
                max_v = max(max_v, v)
            except (IndexError, ValueError):
                pass
    return max_v


def save_prompt_version(run_dir: Path, version: int, text: str) -> Path:
    path = get_prompt_version_path(run_dir, version)
    path.write_text(text, encoding="utf-8")
    return path


def load_prompt_version(run_dir: Path, version: int) -> str:
    path = get_prompt_version_path(run_dir, version)
    if not path.exists():
        print(f"Error: prompt version not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def resolve_prompt(run_dir: Path, prompt_version: int | None) -> tuple[str, int]:
    """
    Return (prompt_text, version_number) based on the requested version
    or the latest available version.
    """
    if prompt_version is not None:
        return load_prompt_version(run_dir, prompt_version), prompt_version
    latest = get_latest_prompt_version(run_dir)
    if latest < 0:
        print("Error: no prompt versions found in run directory.", file=sys.stderr)
        sys.exit(1)
    return load_prompt_version(run_dir, latest), latest


def list_prompt_versions(run_dir: Path) -> list[dict]:
    """Return a sorted list of {version, path} dicts for display."""
    versions = []
    for f in sorted(run_dir.iterdir()):
        if f.is_file() and f.name.startswith("prompt_v") and f.name.endswith(".txt"):
            try:
                v = int(f.stem.split("_v")[1])
                versions.append({"version": v, "path": f})
            except (IndexError, ValueError):
                pass
    return sorted(versions, key=lambda x: x["version"])


# ── Commands ─────────────────────────────────────────────────────────

def cmd_iterate(args):
    client = init_client()
    task_name = args.task
    task_config = TASKS[task_name]

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
        prompt_text, used_version = resolve_prompt(run_dir, args.prompt_version)
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    elif args.seed:
        seed_path = Path(args.seed)
        if not seed_path.is_absolute():
            seed_path = Path(__file__).resolve().parent / seed_path
        if not seed_path.exists():
            print(f"Error: seed prompt not found: {seed_path}", file=sys.stderr)
            sys.exit(1)
        prompt_text = seed_path.read_text(encoding="utf-8")
        run_dir = create_run_dir(task_name)
        run_config = {
            "task": task_name,
            "model": MODEL,
            "seed_prompt": str(seed_path),
            "val_jsonl": task_config["val_jsonl"],
            "test_jsonl": task_config["test_jsonl"],
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2), encoding="utf-8"
        )
        save_prompt_version(run_dir, 0, prompt_text)
        used_version = 0
    else:
        print("Error: provide --seed for a new run or --run-dir to continue", file=sys.stderr)
        sys.exit(1)

    iteration_num = get_latest_iteration(run_dir) + 1
    iter_dir = run_dir / f"iteration_{iteration_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Also save a copy inside the iteration dir for easy reference
    (iter_dir / "prompt_used.txt").write_text(prompt_text, encoding="utf-8")

    print(f"{'=' * 60}")
    print(f"Run: {run_dir.name}")
    print(f"Task: {task_name}  |  Model: {MODEL}  |  Iteration: {iteration_num}")
    print(f"Using prompt: prompt_v{used_version}.txt")
    print(f"{'=' * 60}")

    # Step 2: Evaluate on validation set
    print(f"\n--- Step 2: Evaluate on validation set ---")
    val_jsonl = task_config["val_jsonl"]
    metrics = run_eval(
        client=client,
        task_config=task_config,
        model=MODEL,
        prompt_text=prompt_text,
        jsonl_path=val_jsonl,
        output_dir=iter_dir,
    )

    # Step 3: Analyze misclassifications
    print(f"\n--- Step 3: Analyze misclassifications ---")
    diff_entries = run_diff_analysis(
        client=client,
        model=MODEL,
        eval_results_path=iter_dir / "eval_results.jsonl",
        meta_prompt_path=PROMPTS_DIR / "diff_analysis.txt",
        output_dir=iter_dir,
    )

    if not diff_entries:
        print("\nPerfect score — no errors to analyze. Run 'test' when ready.")
        return

    # Step 4: Cluster failure patterns
    print(f"\n--- Step 4: Cluster failure patterns ---")
    run_clustering(
        client=client,
        model=MODEL,
        diff_explanations_path=iter_dir / "diff_explanations.jsonl",
        meta_prompt_path=PROMPTS_DIR / "clustering.txt",
        output_dir=iter_dir,
    )

    # Step 5: Generate updated prompt → save as next version
    print(f"\n--- Step 5: Generate updated prompt ---")
    updated_text = run_prompt_update(
        client=client,
        model=MODEL,
        current_prompt=prompt_text,
        clusters_path=iter_dir / "clusters.json",
        metrics_path=iter_dir / "metrics.json",
        meta_prompt_path=PROMPTS_DIR / "prompt_update.txt",
        output_dir=iter_dir,
    )

    next_version = get_latest_prompt_version(run_dir) + 1
    version_path = save_prompt_version(run_dir, next_version, updated_text)

    # Show all versions
    versions = list_prompt_versions(run_dir)
    print(f"\n{'=' * 60}")
    print(f"Iteration {iteration_num} complete.")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Errors analyzed: {len(diff_entries)}")
    print(f"\nPrompt versions:")
    for v in versions:
        marker = " <-- NEW" if v["version"] == next_version else ""
        marker = " <-- seed" if v["version"] == 0 else marker
        print(f"  prompt_v{v['version']}.txt{marker}")
    print(f"\nReview/edit the new prompt at:")
    print(f"  {version_path}")
    print(f"\nNext steps:")
    print(f"  Iterate again:    python pipeline.py iterate --task {task_name} --run-dir {run_dir}")
    print(f"  Use specific ver: python pipeline.py iterate --task {task_name} --run-dir {run_dir} --prompt-version {next_version}")
    print(f"  Run on test:      python pipeline.py test --task {task_name} --run-dir {run_dir}")
    print(f"{'=' * 60}")


def cmd_test(args):
    client = init_client()
    task_name = args.task
    task_config = TASKS[task_name]
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    prompt_text, used_version = resolve_prompt(run_dir, args.prompt_version)

    test_dir = run_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "prompt_used.txt").write_text(prompt_text, encoding="utf-8")

    print(f"{'=' * 60}")
    print(f"TEST RUN: {run_dir.name}")
    print(f"Task: {task_name}  |  Model: {MODEL}")
    print(f"Using prompt: prompt_v{used_version}.txt")
    print(f"{'=' * 60}")

    print(f"\n--- Evaluating on test set ---")
    test_jsonl = task_config["test_jsonl"]
    metrics = run_eval(
        client=client,
        task_config=task_config,
        model=MODEL,
        prompt_text=prompt_text,
        jsonl_path=test_jsonl,
        output_dir=test_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"Test complete.")
    print(f"  Prompt used: prompt_v{used_version}.txt")
    print(f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Results: {test_dir / 'eval_results.jsonl'}")
    print(f"  Metrics: {test_dir / 'metrics.json'}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Prompt iteration pipeline for satellite image classification"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # iterate subcommand
    iter_parser = subparsers.add_parser("iterate", help="Run one iteration on the validation set")
    iter_parser.add_argument("--task", required=True, choices=list(TASKS.keys()),
                             help="Task to evaluate (aaa or df)")
    iter_parser.add_argument("--seed", help="Path to seed prompt file (for first iteration)")
    iter_parser.add_argument("--run-dir", help="Path to existing run directory (to continue)")
    iter_parser.add_argument("--prompt-version", type=int, default=None,
                             help="Use a specific prompt version number (default: latest)")
    iter_parser.set_defaults(func=cmd_iterate)

    # test subcommand
    test_parser = subparsers.add_parser("test", help="Run evaluation on the test set")
    test_parser.add_argument("--task", required=True, choices=list(TASKS.keys()),
                             help="Task to evaluate (aaa or df)")
    test_parser.add_argument("--run-dir", required=True,
                             help="Path to the run directory")
    test_parser.add_argument("--prompt-version", type=int, default=None,
                             help="Use a specific prompt version number (default: latest)")
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
