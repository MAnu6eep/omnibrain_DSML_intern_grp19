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
    2. Qwen2.5-VL (OpenRouter)
    3. GPT-4o Mini (OpenRouter)
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

    def build_prompt(self) -> str:
        return """
You are an expert financial document analyst specializing in extracting structured numerical information from charts and tables.

Analyze the provided image carefully.

If the image contains a financial chart (bar chart, line chart, pie chart, stacked bar chart, histogram, or similar), extract every visible numerical value.

Return ONLY valid JSON.

Schema:

{
  "chart_type": "",
  "title": "",
  "x_axis": "",
  "y_axis": "",
  "x_unit": "",
  "y_unit": "",
  "values": [
  {
    "series": "",
    "label": "",
    "value": 0
  }
],
  "summary": ""
}

Instructions:

1. Detect the chart type.
2. Read the chart title.
3. Read X-axis label.
4. Read Y-axis label.
5. Detect units if present.
6. Extract EVERY bar/line value.
7. Preserve original labels.
8. Convert values into numeric format only.
9. Do NOT estimate values unless clearly visible.
10. Ignore decorative graphics.

Return ONLY JSON.

If multiple series exist, include all values.

If no chart is present return:

{
  "chart_type":"unknown",
  "title":"",
  "x_axis":"",
  "y_axis":"",
  "x_unit":"",
  "y_unit":"",
  "values":[],
  "summary":"No financial chart detected."
}
"""

    def _encode_image(self, image_path: str):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _clean_json(self, text: str):

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        return json.loads(text)

    #########################################################
    # GEMINI
    #########################################################

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

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        return self._clean_json(content)

    #########################################################
    # OPENROUTER
    #########################################################

    def _analyze_with_openrouter(self, image_path: str, model: str):

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

        return self._clean_json(content)

    #########################################################
    # MAIN ENTRY
    #########################################################

    def analyze_image(self, image_path: str):

        # 1. Gemini
        try:
            print("Trying Gemini Vision...")
            return self._analyze_with_gemini(image_path)

        except Exception as e:
            print("Gemini failed. Falling back to Qwen...")

        # 2. Qwen
        try:
            print("Trying Qwen2.5-VL...")

            return self._analyze_with_openrouter(
                image_path,
                "qwen/qwen2.5-vl-72b-instruct",
            )

        except Exception:
            print("Qwen failed. Falling back to GPT-4o Mini...")

        # 3. GPT-4o Mini
        try:
            print("Trying GPT-4o Mini...")

            return self._analyze_with_openrouter(
                image_path,
                "openai/gpt-4o-mini",
            )

        except Exception:
            print("GPT-4o Mini failed.")

        return {
            "chart_type": "unknown",
            "title": "",
            "x_axis": "",
            "y_axis": "",
            "values": [],
            "summary": "All providers failed.",
        }