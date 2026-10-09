#!/usr/bin/env python3
"""
Create a 70/30 val/test split for AAA and Double Fences eval datasets.
Pools train + val + test, shuffles with fixed seed, and writes new files.

Train images are fetched from GitHub via the API since they have no local JSONL.
"""
import json
import random
import urllib.request
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
GITHUB_REPO = "selenasun1618/IMINT-Images"
GITHUB_API = "https://api.github.com/repos"
GITHUB_RAW = "https://github.com/selenasun1618/IMINT-Images/blob/main"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def fetch_github_folder(repo_path):
    """Return list of image filenames in a GitHub repo folder via the API."""
    url = f"{GITHUB_API}/{GITHUB_REPO}/contents/{repo_path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        entries = json.loads(resp.read())
    return [e["name"] for e in entries if Path(e["name"]).suffix.lower() in IMAGE_EXTS]


def build_train_items(yes_folder, no_folder, label_field, github_dir):
    """Fetch train image lists from GitHub and return JSONL-style records."""
    items = []
    for folder, label in [(yes_folder, "yes"), (no_folder, "no")]:
        filenames = fetch_github_folder(f"{github_dir}/{folder}")
        print(f"  Fetched {len(filenames)} images from {folder}")
        for name in filenames:
            items.append({
                "item": {
                    label_field: label,
                    "image_name": name,
                    "image_url": f"{GITHUB_RAW}/{github_dir}/{folder}/{name}?raw=true",
                }
            })
    return items


def pool_jsonl(*paths):
    items = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def split_70_30(items, seed=RANDOM_SEED):
    r = random.Random(seed)
    shuffled = items[:]
    r.shuffle(shuffled)
    n_val = round(len(shuffled) * 0.70)
    return shuffled[:n_val], shuffled[n_val:]


def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} items → {path}")


def main():
    # ── AAA ──────────────────────────────────────────────────────────────
    print("=== AAA ===")
    print("  Fetching train images from GitHub...")
    aaa_train = build_train_items(
        yes_folder="yes_aaa_train",
        no_folder="no_aaa_train",
        label_field="aaa_present",
        github_dir="AAA",
    )
    aaa_existing = pool_jsonl(
        EVALS_DIR / "aaa_eval_val.jsonl",
        EVALS_DIR / "aaa_eval_test.jsonl",
    )
    aaa_all = aaa_train + aaa_existing
    print(f"  Total pooled: {len(aaa_all)} (train={len(aaa_train)}, existing val+test={len(aaa_existing)})")
    yes = sum(1 for x in aaa_all if x["item"]["aaa_present"] == "yes")
    no  = sum(1 for x in aaa_all if x["item"]["aaa_present"] == "no")
    print(f"  yes={yes}, no={no}")

    aaa_val, aaa_test = split_70_30(aaa_all)
    print(f"  → val={len(aaa_val)}, test={len(aaa_test)}")

    write_jsonl(EVALS_DIR / "aaa_eval_val.jsonl",  aaa_val)
    write_jsonl(EVALS_DIR / "aaa_eval_test.jsonl", aaa_test)

    # ── Double Fences ─────────────────────────────────────────────────────
    print("\n=== Double Fences ===")
    print("  Fetching train images from GitHub...")
    df_train = build_train_items(
        yes_folder="yes_df_train",
        no_folder="no_df_train",
        label_field="double_fences_present",
        github_dir="Double-Fences",
    )
    df_existing = pool_jsonl(
        EVALS_DIR / "double_fences_eval_with_explanations_val.jsonl",
        EVALS_DIR / "double_fences_eval_with_explanations_test.jsonl",
    )
    df_all = df_train + df_existing
    print(f"  Total pooled: {len(df_all)} (train={len(df_train)}, existing val+test={len(df_existing)})")
    yes = sum(1 for x in df_all if x["item"]["double_fences_present"] == "yes")
    no  = sum(1 for x in df_all if x["item"]["double_fences_present"] == "no")
    print(f"  yes={yes}, no={no}")

    df_val, df_test = split_70_30(df_all)
    print(f"  → val={len(df_val)}, test={len(df_test)}")

    write_jsonl(EVALS_DIR / "double_fences_eval_with_explanations_val.jsonl",  df_val)
    write_jsonl(EVALS_DIR / "double_fences_eval_with_explanations_test.jsonl", df_test)

    print("\nDone.")


if __name__ == "__main__":
    main()
