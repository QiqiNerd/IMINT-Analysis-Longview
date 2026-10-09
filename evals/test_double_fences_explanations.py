import json
from openai import OpenAI
from df_prompts_with_explanations import ZERO_SHOT_PROMPT_WITH_EXPLANATIONS

# OpenAI reads OPENAI_API_KEY from the environment.
client = OpenAI()

def test_double_fences_with_explanations():
    """Test the model with a sample image to verify explanation output"""

    # Test with a known double fences image
    test_image_url = "https://github.com/selenasun1618/IMINT-Images/blob/main/Double-Fences/df_train/Ling_ao-1_M310_22.60184_114.55039_0.5km.png?raw=true"

    # Adapt the prompt for this specific image
    adapted_prompt = ZERO_SHOT_PROMPT_WITH_EXPLANATIONS.replace("{{ item.image_url }}", "")

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "double_fences_with_explanation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "double_fences_present": {"type": "string", "enum": ["yes", "no"]},
                        "explanation": {"type": "string"}
                    },
                    "required": ["double_fences_present", "explanation"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        messages=[
            {
                "role": "developer",
                "content": "You are an expert satellite analyst responsible for informing the U.S. government about the status of nuclear power plants. You are looking for double fences around buildings, which is an indicator that it's a nuclear site. You are to decide whether the given image contains double fences or not and provide a detailed explanation."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": adapted_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": test_image_url}
                    }
                ]
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)

    print("🧪 Testing Double Fences with Explanations")
    print(f"Image: {test_image_url}")
    print(f"Prediction: {result['double_fences_present']}")
    print(f"Explanation: {result['explanation']}")
    print()

    # Test with a non-double fences image
    test_image_url_no = "https://github.com/selenasun1618/IMINT-Images/blob/main/Double-Fences/no_df_train/XIAN_34.43031_108.93112_0.5km.png?raw=true"

    response_no = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "double_fences_with_explanation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "double_fences_present": {"type": "string", "enum": ["yes", "no"]},
                        "explanation": {"type": "string"}
                    },
                    "required": ["double_fences_present", "explanation"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        messages=[
            {
                "role": "developer",
                "content": "You are an expert satellite analyst responsible for informing the U.S. government about the status of nuclear power plants. You are looking for double fences around buildings, which is an indicator that it's a nuclear site. You are to decide whether the given image contains double fences or not and provide a detailed explanation."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": adapted_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": test_image_url_no}
                    }
                ]
            }
        ]
    )

    result_no = json.loads(response_no.choices[0].message.content)

    print("🧪 Testing Non-Double Fences with Explanations")
    print(f"Image: {test_image_url_no}")
    print(f"Prediction: {result_no['double_fences_present']}")
    print(f"Explanation: {result_no['explanation']}")
    print()

    print("✅ Test completed! Both responses include explanations as expected.")

if __name__ == "__main__":
    test_double_fences_with_explanations()
