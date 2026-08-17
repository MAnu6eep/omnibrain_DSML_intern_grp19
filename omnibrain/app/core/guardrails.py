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
    # Harmful / violent requests
    r"\bhow\s+(to|can\s+i|do\s+i)\s+(hack|attack|kill)\b",
    r"\bhow\s+(to|can\s+i|do\s+i)\s+(make|build|create)\s+(a\s+)?(bomb|explosive)s?\b",
    r"\bhow\s+(to|can\s+i|do\s+i)\s+(make|build|create)\s+(a\s+)?(weapon|poison)s?\b",
    r"\bgive\s+me\s+(instructions|steps)\s+to\s+(attack|kill|hack)\b",
    r"\b(instructions|steps)\s+for\s+(making|creating)\s+(a\s+)?(bomb|explosive|weapon)\b",

    # Authentication / security bypass
    r"\b(bypass|disable|circumvent)\s+(the\s+)?(authentication|login|login\s+system)\b",
    r"\b(bypass|disable|circumvent)\s+(the\s+)?(guardrails?|safety\s+controls?|security\s+controls?)\b",
    r"\bhow\s+(to|can\s+i|do\s+i)\s+(bypass|disable|circumvent)\b",

    # Prompt/safety manipulation
    r"\b(ignore|disregard)\s+(all\s+)?(previous|prior)\s+instructions\b",
    r"\bact\s+as\s+an\s+unrestricted\s+assistant\b",
    r"\bhow to build an explosive\b",
    r"\binstructions for making explosives\b",
    r"\bdisable the security checks\b",
    r"\bsecurity checks\b",
    r"\binvestment advice\b",
    r"\bignore the document domain\b",
    r"\bignore the safety policy\b",
    r"\bharmful instructions\b",
    r"\breveal information outside\b",
    r"\boutside the supported document domain\b",
    r"\bhow\s+do\s+i\s+build\s+(an?\s+)?explosive\b",

]


_OFF_TOPIC_PATTERNS = [
    r"\brecipe\b",
    r"\bfootball\b",
    r"\bsoccer\b",
    r"\bcricket\b",
    r"\bcricket\s+score\b",
    r"\bmovie\s+recommendation\b",
    r"\bplay\s+a\s+game\b",
    r"\bcapital\s+of\s+france\b",
    r"\bfifa\s+world\s+cup\b",
    r"\btell\s+me\s+a\s+joke\b",
    r"\blatest\s+news\b",
    r"\bpresident\s+of\s+the\s+united\s+states\b",
    r"\bbest\s+smartphone\b",
    r"\bplan\s+(a\s+)?vacation\b",
    r"\bweather\s+today\b",
    r"\bcryptocurrency\b",
    r"\bstock\s+(investment|recommendation)\b",
    r"\bwhich\s+stock\s+should\s+i\s+buy\b",
    r"\btoday'?s\s+sports\s+results\b",
    r"\bwrite\s+me\s+a\s+romantic\s+story\b",
    r"\bprogramming\s+joke\b",
    r"\brecommend\s+(a\s+)?movie\b",
]


_BLOCKED_OUTPUT_PATTERNS = [
    # Harmful instructions
    r"\bhow\s+to\s+(hack|attack|kill)\b",
    r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive)s?\b",
    r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?(weapon|poison)s?\b",
    r"\b(instructions|steps)\s+(for|to)\s+(making|creating|building)\s+(a\s+)?(bomb|explosive|weapon)\b",

    # Security bypass
    r"\bhow\s+to\s+(bypass|disable|circumvent)\b",
    r"\b(bypass|disable|circumvent)\s+(the\s+)?(authentication|login|guardrails?|safety\s+controls?|security\s+controls?)\b",

    r"\bhow to make an explosive\b",
    r"\bhow to make explosives\b",

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