import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from aaa_prompts import *

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_jsonl_file(jsonl_path, dataset):
    """Create a JSONL file with the image and AAA presence."""
    github_url = f"https://github.com/selenasun1618/IMINT-Images/blob/main/AAA/aaa_{dataset}_combined/"
    local_dir = "../IMINT-Images/AAA/"
    AAA_local_folder = f"yes_aaa_{dataset}/"
    Non_AAA_local_folder = f"no_aaa_{dataset}/"

    total_written = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:  # Overwrite on each run
        # Process AAA images
        aaa_path = os.path.join(local_dir, AAA_local_folder)
        for img_name in os.listdir(aaa_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                record = {
                    "item": {
                        "aaa_present": "yes",
                        "image_name": img_name,
                        "image_url": f"{github_url}{img_name}?raw=true",
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

        # Process Non-AAA images
        non_aaa_path = os.path.join(local_dir, Non_AAA_local_folder)
        for img_name in os.listdir(non_aaa_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                record = {
                    "item": {
                        "aaa_present": "no",
                        "image_name": img_name,
                        "image_url": f"{github_url}{img_name}?raw=true",
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"✅ JSONL file created at '{jsonl_path}' with {total_written} entries.")


def upload_files(jsonl_path):
    file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="evals"
    )
    return file

def create_eval(name):
    eval_obj = client.evals.create(
        name=name,
        data_source_config={
            "type": "custom",
            "item_schema": {
                "type": "object",
                "properties": {
                    "aaa_present": {"type": "string"},
                    "image_name": {"type": "string"},
                    "image_url": {"type": "string"},
                },
                "required": ["image_url", "aaa_present", "image_name"],
            },
            "include_sample_schema": True
        },
        testing_criteria=[
            {
                "name": "AAA classification grader",
                "type": "string_check",
                "input": "{{ sample.output_json.aaa_present }}",
                "operation": "eq",
                "reference": "{{ item.aaa_present }}"
            }
        ],
    )
    return eval_obj


def run_eval(eval_id, file_id, user_prompt=PROMPT, model="gpt-5.4"):
    """Run the eval with the given ID and file path."""

    developer_prompt = """
    CRITICAL: You MUST respond with ONLY a valid JSON object. NO markdown, NO code blocks, NO explanations outside the JSON.

    You are an expert nuclear analyst responsible for informing the U.S. government about the status of nuclear power plants.
    You are looking for anti-aircraft artillery (AAA) sites, which are indicators of nuclear power plants nearby.
    You are to decide whether the given image contains an AAA site or not and provide a detailed explanation.

    REQUIRED FORMAT:
    {"aaa_present": "yes", "explanation": "your detailed analysis here"}
    OR
    {"aaa_present": "no", "explanation": "your detailed analysis here"}

    DO NOT include:
    - Markdown formatting (```json, ```, etc.)
    - Any text before or after the JSON
    - Code block delimiters
    - Additional commentary

    ONLY return the raw JSON object as shown above.
    """

    eval_run = client.evals.runs.create(
        eval_id=eval_id,
        name="AAA Eval Run",
        data_source={
            "type": "completions",
            "model": model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "aaa_with_explanation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "aaa_present": {"type": "string", "enum": ["yes", "no"]},
                            "explanation": {"type": "string"}
                        },
                        "required": ["aaa_present", "explanation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            "source": {"type": "file_id", "id": file_id},
            "input_messages": {
                "type": "template",
                "template": [
                    {
                        "role": "developer",
                        "content": developer_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_prompt.replace("{{ item.image_url }}", "")
                            },
                            {
                                "type": "input_image",
                                "image_url": "{{ item.image_url }}",
                                "detail": "auto"
                            }
                        ]
                    }
                ]
            }
        }
    )

    return eval_run

def main():

    dataset = "test" # "test" or "val"
    # 1. Create JSONL file
    # jsonl_path = Path(f"evals/aaa_eval_{dataset}.jsonl").resolve()
    # print(f"Creating JSONL file at: {jsonl_path}")
    # create_jsonl_file(jsonl_path, dataset)
    # file = upload_files(jsonl_path=jsonl_path)
    # print(f"Jsonl file uploaded: {file.id}")

    # 2. Create the eval
    eval_obj = create_eval(name="[Test] AAA - FT NO Guidance")
    print(f"Eval created: {eval_obj.id}")

    file_id = os.getenv("OPENAI_EVAL_FILE_ID")
    if not file_id:
        raise SystemExit("Set OPENAI_EVAL_FILE_ID to a file available to your account.")
    # 4. Run the eval
    model = os.getenv("OPENAI_EVAL_MODEL", "gpt-4o")
    eval_run = run_eval(eval_id=eval_obj.id, file_id=file_id, user_prompt=PROMPT, model=model)
    print(f"Eval run started: {eval_run.id}")

    run = client.evals.runs.retrieve(eval_id=eval_obj.id, run_id=eval_run.id)
    print(f"Eval run status: {run.status}")

if __name__ == "__main__":
    main()
