import re
import base64
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


class VLMReasoner:
    """
    Vision Language Model Reasoner

    Provider order:
    1. Gemini Vision
    2. Qwen2.5-VL via OpenRouter
    3. GPT-4o Mini via OpenRouter
    """

    def __init__(self):
        gemini_key = os.getenv("GEMINI_API_KEY")

        self.llm = None

        if gemini_key:
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                google_api_key=gemini_key,
                temperature=0,
                max_retries=0,
            )

    # =========================================================
    # PROMPT
    # =========================================================

    def build_prompt(self) -> str:
        return """
Analyze the supplied chart or table image.

Extract only information that is visibly present in the image.

Return ONLY valid JSON using this structure:

{
    "chart_type": "",
    "title": "",
    "x_axis": "",
    "y_axis": "",
    "values": [
        {
            "label": "",
            "value": ""
        }
    ],
    "summary": ""
}

STRICT VISUAL SAFETY RULES:

1. Never invent or estimate numerical values.
2. Only include numerical values that are explicitly visible in the image.
3. Do not calculate missing values.
4. Do not infer values from the shape or height of a bar.
5. Do not use numbers from external knowledge.
6. If a numerical value is unclear or not visible, omit it.
7. Preserve the exact numerical value visible in the image.
8. If no reliable numerical values are visible, return an empty "values" list.
9. Do not include Markdown or code fences.
"""

    # =========================================================
    # IMAGE ENCODING
    # =========================================================

    def _encode_image(self, image_path: str):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # =========================================================
    # JSON CLEANING
    # =========================================================

    def _clean_json(self, text: str):
        if not isinstance(text, str):
            text = str(text)

        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        return json.loads(text)

    # =========================================================
    # GEMINI
    # =========================================================

    def _analyze_with_gemini(self, image_path: str):

        if self.llm is None:
            raise Exception("Gemini not configured.")

        image_data = self._encode_image(image_path)

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": self.build_prompt(),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    },
                },
            ]
        )

        response = self.llm.invoke([message])

        content = response.content

        if isinstance(content, str):
            response_text = content
        else:
            response_text = ""

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            response_text += item.get("text", "")
                        elif "text" in item:
                            response_text += item.get("text", "")
                    else:
                        response_text += str(item)
            else:
                response_text = str(content)

        return self._clean_json(response_text)

    # =========================================================
    # OPENROUTER
    # =========================================================

    def _analyze_with_openrouter(
        self,
        image_path: str,
        model: str,
    ):

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise Exception("OPENROUTER_API_KEY missing.")

        image_data = self._encode_image(image_path)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.build_prompt(),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=90,
        )

        response.raise_for_status()

        data = response.json()

        content = data["choices"][0]["message"]["content"]

        if isinstance(content, str):
            response_text = content
        else:
            response_text = str(content)

        return self._clean_json(response_text)

    # =========================================================
    # VALIDATION / GUARDRAIL
    # =========================================================

    def _validate_result(self, result):

        if not isinstance(result, dict):
            return {
                "chart_type": "unknown",
                "title": "",
                "x_axis": "",
                "y_axis": "",
                "values": [],
                "summary": "VLM output blocked: invalid response format.",
                "guardrail_status": "blocked",
            }

        values = result.get("values", [])

        if not isinstance(values, list):
            values = []

        validated_values = []

        for item in values:

            if not isinstance(item, dict):
                continue

            label = item.get("label")
            value = item.get("value")

            if label is None or value is None:
                continue

            value_text = str(value).strip()

            # Numerical values must contain an explicitly
            # returned numeric value.
            if not re.search(r"\d", value_text):
                continue

            validated_values.append(
                {
                    "label": str(label),
                    "value": value,
                }
            )

        result["values"] = validated_values

        result["guardrail_status"] = "passed"

        return result

    # =========================================================
    # MAIN ENTRY
    # =========================================================

    def analyze_image(self, image_path: str):

        # -----------------------------------------------------
        # 1. GEMINI
        # -----------------------------------------------------

        try:
            print("Trying Gemini Vision...")

            result = self._analyze_with_gemini(image_path)

            return self._validate_result(result)

        except Exception as e:
            print(f"Gemini failed: {e}")
            print("Falling back to Qwen...")

        # -----------------------------------------------------
        # 2. QWEN
        # -----------------------------------------------------

        try:
            print("Trying Qwen2.5-VL...")

            result = self._analyze_with_openrouter(
                image_path,
                "qwen/qwen2.5-vl-72b-instruct",
            )

            return self._validate_result(result)

        except Exception as e:
            print(f"Qwen failed: {e}")
            print("Falling back to GPT-4o Mini...")

        # -----------------------------------------------------
        # 3. GPT-4o MINI
        # -----------------------------------------------------

        try:
            print("Trying GPT-4o Mini...")

            result = self._analyze_with_openrouter(
                image_path,
                "openai/gpt-4o-mini",
            )

            return self._validate_result(result)

        except Exception as e:
            print(f"GPT-4o Mini failed: {e}")

        # -----------------------------------------------------
        # ALL PROVIDERS FAILED
        # -----------------------------------------------------

        return {
            "chart_type": "unknown",
            "title": "",
            "x_axis": "",
            "y_axis": "",
            "values": [],
            "summary": "All providers failed.",
            "guardrail_status": "blocked",
        }