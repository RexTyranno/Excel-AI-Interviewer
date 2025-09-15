import os
import json
from typing import Any, Dict, Optional


def _openai_client():
    try:
        # Lazy import; return None if no key
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        from openai import OpenAI  # type: ignore
        return OpenAI()
    except Exception:
        return None


def paraphrase_question(
    asked: Dict[str, Any],
    *,
    seed: str = "",
    session_summary: Optional[Dict[str, Any]] = None,
    chat_window: Optional[list] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Return an LLM-paraphrased question or None if unavailable.

    - Preserves difficulty/intent; avoids revealing solutions.
    - Accepts optional lightweight context for better phrasing.
    """
    client = _openai_client()
    if not client:
        return None

    payload = {
        "question": {
            "id": asked.get("id"),
            "text": asked.get("text"),
            "topic": asked.get("topic"),
            "tier": asked.get("tier"),
            "level_tag": asked.get("level_tag"),
            "format": asked.get("question_format"),
        },
        "seed": seed,
        "session_summary": session_summary or {},
        "recent_chat": chat_window[-6:] if chat_window else [],
    }

    sys = (
        "You are a professional interviewer. Rephrase the question so it sounds natural and fresh, "
        "WITHOUT changing the intent, constraints, or difficulty. Do not give away the solution. "
        "Keep it concise (1–2 sentences). Use the provided context to maintain consistency."
    )

    try:
        resp = client.chat.completions.create(
            model=model or os.environ.get("PARAPHRASE_MODEL", "gpt-4o-mini"),
            temperature=0.4,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or None
    except Exception:
        return None


def clarify(
    asked: Dict[str, Any],
    user_msg: str,
    *,
    session_summary: Optional[Dict[str, Any]] = None,
    chat_window: Optional[list] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Return a brief clarification response, declining to reveal solutions."""
    client = _openai_client()
    if not client:
        return None

    payload = {
        "question": {
            "id": asked.get("id"),
            "text": asked.get("text"),
            "topic": asked.get("topic"),
            "tier": asked.get("tier"),
            "level_tag": asked.get("level_tag"),
            "format": asked.get("question_format"),
        },
        "candidate_msg": user_msg,
        "session_summary": session_summary or {},
        "recent_chat": chat_window[-6:] if chat_window else [],
    }

    sys = (
        "Act as an interviewer clarifying the task without disclosing the solution. "
        "Be helpful and concise (<= 2 sentences). If asked for the answer, politely decline and restate the goal."
    )

    try:
        resp = client.chat.completions.create(
            model=model or os.environ.get("CLARIFY_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or None
    except Exception:
        return None 