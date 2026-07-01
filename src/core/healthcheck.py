"""Health checks for production deployment verification.

Run as: python -m src.core.healthcheck

Returns JSON to stdout and exits 0 (all passed) or 1 (any failed).
"""

import json
import logging
import sys
import time

from src.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)


def check_chromadb() -> tuple[bool, str]:
    """Check ChromaDB connectivity by initializing client and calling heartbeat."""
    try:
        import chromadb
        from chromadb.config import Settings

        from src.core.config import CHROMA_DIR

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        client.heartbeat()
        return True, "ChromaDB heartbeat OK"
    except Exception as e:
        return False, f"ChromaDB check failed: {e}"


def check_embeddings() -> tuple[bool, str]:
    """Check embedding model loads and produces output."""
    try:
        from src.rag.embeddings import embed_text

        vec = embed_text("healthcheck")
        if vec and len(vec) > 0:
            return True, f"Embedding OK (dim={len(vec)})"
        return False, "Embedding returned empty vector"
    except Exception as e:
        return False, f"Embedding check failed: {e}"


def check_deepseek_api() -> tuple[bool, str]:
    """Check DeepSeek API connectivity (only if API key is configured)."""
    if not DEEPSEEK_API_KEY:
        return True, "DeepSeek API: skipped (no key configured)"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=5)
        client.models.list()
        return True, f"DeepSeek API OK (model={DEEPSEEK_MODEL})"
    except Exception as e:
        return False, f"DeepSeek API check failed: {e}"


def run_checks() -> tuple[bool, dict[str, dict]]:
    """Run all health checks and return (overall_status, details)."""
    checks = {
        "chromadb": check_chromadb(),
        "embeddings": check_embeddings(),
        "deepseek_api": check_deepseek_api(),
    }

    details: dict[str, dict] = {}
    all_ok = True
    for name, (ok, msg) in checks.items():
        details[name] = {"passed": ok, "message": msg}
        if not ok:
            all_ok = False

    return all_ok, details


def main() -> int:
    """Run health checks, print JSON, return exit code."""
    t0 = time.time()
    ok, details = run_checks()
    elapsed = round((time.time() - t0) * 1000)

    result = {
        "status": "ok" if ok else "degraded",
        "duration_ms": elapsed,
        "checks": details,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
