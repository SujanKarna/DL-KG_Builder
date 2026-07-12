import requests
import base64
from dotenv import load_dotenv
import os

load_dotenv()  # load .env file

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

class HFRemoteVLM:
    def __call__(self, prompt, image_paths=None):
        images_encoded = []

        if image_paths:
            for img_path in image_paths:
                with open(img_path, "rb") as f:
                    images_encoded.append(base64.b64encode(f.read()).decode("utf-8"))

        payload = {
            "inputs": {
                "prompt": prompt,
                "images": images_encoded
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()


def get_llm():
    return HFRemoteVLM()
