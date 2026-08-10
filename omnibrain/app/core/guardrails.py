import logging
import re
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig


BASE_DIR = Path(__file__).resolve().parents[3]
GUARDRAILS_DIR = BASE_DIR / "guardrails"

logger = logging.getLogger(__name__)


def _load_guardrails():
    config = RailsConfig.from_path(str(GUARDRAILS_DIR))
    return LLMRails(config)


# Validate that the NeMo Guardrails configuration can be loaded.
rails = _load_guardrails()


# Basic policy terms for requests that are clearly outside
# the supported OmniBrain document/RAG domain.
_BLOCKED_INPUT_PATTERNS = [
    r"\bhow to hack\b",
    r"\bhow to attack\b",
    r"\bhow to kill\b",
    r"\bhow to make (a bomb|explosives?)\b",
    r"\bhow to make (a weapon|poison)\b",
    r"\b(bypass|disable) authentication\b",
    r"\b(bypass|disable) guardrails?\b",
]

_OFF_TOPIC_PATTERNS = [
    r"\brecipe\b",
    r"\bfootball\b",
    r"\bcricket score\b",
    r"\bmovie recommendation\b",
    r"\bplay a game\b",
]
_BLOCKED_OUTPUT_PATTERNS = [
    r"\bhow to (hack|attack|kill)\b",
    r"\bmake (a bomb|explosives?)\b",
    r"\bhow to make (a weapon|poison)\b",
]

def _matches_any(message: str, patterns: list[str]) -> bool:
    text = message.lower().strip()

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


async def check_input(message: str) -> bool:
    """
    Validate user input before it reaches the agent pipeline.

    Returns:
        True  -> request is allowed
        False -> request must be rejected
    """

    if not message or not message.strip():
        return False

    if _matches_any(message, _BLOCKED_INPUT_PATTERNS):
        logger.warning(
            "Input guardrail blocked unsafe request"
        )
        return False

    if _matches_any(message, _OFF_TOPIC_PATTERNS):
        logger.warning(
            "Input guardrail blocked off-topic request"
        )
        return False

    logger.info(
        "Input guardrail passed"
    )

    return True


async def check_output(response: str) -> bool:
    """
    Validate generated assistant output.
    """

    if not response:
        logger.warning("Output guardrail rejected empty response")
        return False

    # Make sure the generated response is always treated as text.
    response = str(response).strip()

    if not response:
        logger.warning("Output guardrail rejected blank response")
        return False

    logger.info(
        "Output guardrail checking response: %s",
        response[:300],
    )

    if _matches_any(response, _BLOCKED_OUTPUT_PATTERNS):
        logger.warning(
            "Output guardrail blocked unsafe response: %s",
            response[:300],
        )
        return False

    logger.info("Output guardrail passed")
    return True