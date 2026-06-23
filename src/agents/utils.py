"""Shared utilities for agent outputs — JSON extraction and retry logic."""

import json
import time
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[Union[dict, list]]:
    """Extract JSON from LLM output, handling conversational prefixes."""
    if not text:
        return None

    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { and last } as a last resort
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try finding array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def invoke_with_retry(agent_executor, input_dict: dict, max_retries: int = 3, base_delay: float = 2.0) -> dict:
    """Invoke a LangChain AgentExecutor with exponential-backoff retry.

    Handles transient failures: rate limits, server errors, timeouts, connection
    drops, and occasional LLM JSON formatting glitches.  Non-retryable errors
    (401 Unauthorized / invalid API key) are re-raised immediately.

    Args:
        agent_executor: A LangChain AgentExecutor instance.
        input_dict: The input dict to pass to agent_executor.invoke().
        max_retries: Max retry attempts (default 3, i.e. 4 total tries).
        base_delay: Initial backoff delay in seconds (doubles each attempt).

    Returns:
        Parsed JSON dict from the agent output, or an error dict on exhaustion.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = agent_executor.invoke(input_dict)
            output = result["output"]
            parsed = extract_json(output)
            if parsed:
                return parsed
            # LLM output was valid text but not parseable JSON — worth a retry
            raise ValueError("Agent returned unparseable JSON output")
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            # Never retry authentication errors
            if any(kw in msg for kw in ("401", "unauthorized", "invalid_api_key", "invalid api key")):
                raise
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Agent invoke failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, max_retries + 1, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error("Agent invoke exhausted %d retries: %s", max_retries + 1, e)

    return {"error": "API 调用失败，请稍后重试", "detail": str(last_error)}
