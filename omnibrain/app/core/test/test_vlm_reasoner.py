from unittest.mock import patch

from omnibrain.app.services.vision.vlm_reasoner import VLMReasoner


def test_vlm_fallback_when_gemini_fails():
    reasoner = VLMReasoner()

    qwen_result = {
        "chart_type": "bar",
        "title": "Test Chart",
        "x_axis": "X",
        "y_axis": "Y",
        "values": [],
        "summary": "Qwen result",
    }

    with patch.object(
        reasoner,
        "_analyze_with_gemini",
        side_effect=Exception("Gemini failed"),
    ), patch.object(
        reasoner,
        "_analyze_with_openrouter",
        return_value=qwen_result,
    ) as openrouter:

        result = reasoner.analyze_image("test-image.png")

    assert result["chart_type"] == "bar"
    assert openrouter.call_count == 1
    assert openrouter.call_args.args[1] == "qwen/qwen2.5-vl-72b-instruct"


def test_vlm_fallback_when_qwen_fails():
    reasoner = VLMReasoner()

    gpt_result = {
        "chart_type": "line",
        "title": "Fallback Chart",
        "x_axis": "X",
        "y_axis": "Y",
        "values": [],
        "summary": "GPT result",
    }

    with patch.object(
        reasoner,
        "_analyze_with_gemini",
        side_effect=Exception("Gemini failed"),
    ), patch.object(
        reasoner,
        "_analyze_with_openrouter",
        side_effect=[
            Exception("Qwen failed"),
            gpt_result,
        ],
    ) as openrouter:

        result = reasoner.analyze_image("test-image.png")

    assert result["chart_type"] == "line"
    assert openrouter.call_count == 2
    assert openrouter.call_args_list[1].args[1] == "openai/gpt-4o-mini"


def test_vlm_returns_blocked_when_all_providers_fail():
    reasoner = VLMReasoner()

    with patch.object(
        reasoner,
        "_analyze_with_gemini",
        side_effect=Exception("Gemini failed"),
    ), patch.object(
        reasoner,
        "_analyze_with_openrouter",
        side_effect=Exception("Provider failed"),
    ):

        result = reasoner.analyze_image("test-image.png")

    assert result["guardrail_status"] == "blocked"
    assert result["chart_type"] == "unknown"
    assert result["values"] == []
