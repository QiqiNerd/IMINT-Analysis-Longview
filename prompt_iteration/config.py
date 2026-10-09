"""
Task definitions, data paths, and model configuration for the prompt iteration pipeline.

To reconfigure data splits:
  - Edit the val_jsonl / test_jsonl paths below, OR
  - Regenerate JSONL files using evals/aaa_eval.py::create_jsonl_file()
    or evals/double_fences_eval_with_explanations.py::create_jsonl_file()
"""

import os
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"

MODEL = "gpt-5.4"

TASKS = {
    "aaa": {
        "label_field": "aaa_present",
        "val_jsonl": str(EVALS_DIR / "aaa_eval_val.jsonl"),
        "test_jsonl": str(EVALS_DIR / "aaa_eval_test.jsonl"),
        "developer_prompt": (
            "You are an expert nuclear analyst responsible for informing the "
            "U.S. government about the status of nuclear power plants. "
            "You are looking for anti-aircraft artillery (AAA) sites, which "
            "are indicators of nuclear power plants nearby. "
            "You are to decide whether the given image contains an AAA site "
            "or not and provide a detailed explanation.\n\n"
            "CRITICAL: Respond with ONLY a valid JSON object. "
            "No markdown, no code blocks, no text outside the JSON."
        ),
        "response_schema": {
            "name": "classification_result",
            "schema": {
                "type": "object",
                "properties": {
                    "aaa_present": {"type": "string", "enum": ["yes", "no"]},
                    "explanation": {"type": "string"},
                },
                "required": ["aaa_present", "explanation"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    "df": {
        "label_field": "double_fences_present",
        "val_jsonl": str(EVALS_DIR / "double_fences_eval_with_explanations_val.jsonl"),
        "test_jsonl": str(EVALS_DIR / "double_fences_eval_with_explanations_test.jsonl"),
        "developer_prompt": (
            "You are an expert satellite analyst responsible for informing the "
            "U.S. government about the status of nuclear power plants. "
            "You are looking for double fences around buildings, which is an "
            "indicator that it is a nuclear site. "
            "You are to decide whether the given image contains double fences "
            "or not and provide a detailed explanation.\n\n"
            "CRITICAL: Respond with ONLY a valid JSON object. "
            "No markdown, no code blocks, no text outside the JSON."
        ),
        "response_schema": {
            "name": "classification_result",
            "schema": {
                "type": "object",
                "properties": {
                    "double_fences_present": {"type": "string", "enum": ["yes", "no"]},
                    "explanation": {"type": "string"},
                },
                "required": ["double_fences_present", "explanation"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
}

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
