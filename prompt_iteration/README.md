# Prompt Iteration Pipeline

Iteratively improve satellite image classification prompts by evaluating on a validation set, analyzing failure patterns, and generating improved prompts.

## Setup

```bash
# From the project root
source .venv/bin/activate
```

Make sure your `.env` file has a valid `OPENAI_API_KEY`.

## Quick Start (AAA task)

```bash
cd prompt_iteration

# 1. Run first iteration on validation set using the seed prompt
python pipeline.py iterate --task aaa --seed prompts/aaa_seed.txt

# 2. Review the generated prompt (path printed at end of step 1)
#    Edit runs/aaa_<timestamp>/prompt_v1.txt if desired

# 3. Run another iteration (auto-uses latest prompt version)
python pipeline.py iterate --task aaa --run-dir runs/aaa_<timestamp>

# 4. When satisfied, run on test set
python pipeline.py test --task aaa --run-dir runs/aaa_<timestamp>
```

## Quick Start (Double Fences task)

```bash
cd prompt_iteration

python pipeline.py iterate --task df --seed prompts/df_seed.txt
# review runs/df_<timestamp>/prompt_v1.txt
python pipeline.py iterate --task df --run-dir runs/df_<timestamp>
python pipeline.py test --task df --run-dir runs/df_<timestamp>
```

## Prompt Versioning

Prompts are saved as versioned files at the run level:

```
runs/aaa_20260320_143000/
├── prompt_v0.txt    <- seed (never overwritten)
├── prompt_v1.txt    <- generated after iteration 1
├── prompt_v2.txt    <- generated after iteration 2
└── ...
```

The pipeline auto-detects the latest version. Override with `--prompt-version`:

```bash
# Iterate using a specific prompt version
python pipeline.py iterate --task aaa --run-dir runs/aaa_<timestamp> --prompt-version 1

# Test with a specific prompt version
python pipeline.py test --task aaa --run-dir runs/aaa_<timestamp> --prompt-version 2
```

You can edit any `prompt_v*.txt` file in place before continuing.

## Pipeline Steps

Each `iterate` command runs these steps:

| Step | What it does | Output |
|------|-------------|--------|
| 2 | Classify every image in the val set | `eval_results.jsonl`, `metrics.json` |
| 3 | For each wrong classification, ask GPT-5.4 why it was wrong | `diff_explanations.jsonl` |
| 4 | Cluster the failure explanations into categories | `clusters.json` |
| 5 | Rewrite the prompt to address failure patterns | `prompt_v{N}.txt` |

The `test` command runs step 2 only, on the test set.

## Configuring Data Splits

Edit `config.py` to change validation/test JSONL paths:

```python
TASKS = {
    "aaa": {
        "val_jsonl": "...",   # path to validation JSONL
        "test_jsonl": "...",  # path to test JSONL
        ...
    },
}
```

To regenerate JSONL files from local image directories, use the existing functions in `evals/aaa_eval.py` or `evals/double_fences_eval_with_explanations.py`.

## Editing Meta-Prompts

The prompts used by steps 3-5 are plain text files in `prompts/`:

- `diff_analysis.txt` — how to explain a misclassification (step 3)
- `clustering.txt` — how to group failure patterns (step 4)
- `prompt_update.txt` — how to rewrite the classification prompt (step 5)

## Run Directory Structure

```
runs/aaa_20260320_143000/
├── run_config.json
├── prompt_v0.txt
├── prompt_v1.txt
├── iteration_1/
│   ├── prompt_used.txt
│   ├── eval_results.jsonl
│   ├── metrics.json
│   ├── diff_explanations.jsonl
│   ├── clusters.json
│   └── updated_prompt.txt
├── iteration_2/
│   └── ...
└── test/
    ├── prompt_used.txt
    ├── eval_results.jsonl
    └── metrics.json
```
