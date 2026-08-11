from __future__ import annotations

import json
import re
from typing import Any


class ParseError(ValueError):
    """Raised when model output cannot be converted to the public schema."""


_LABELS = {"normal", "under_extrusion", "unsure"}


def parse_model_output(text: str) -> dict[str, Any]:
    data = _load_json_object(text)
    label = data.get("label")
    if label not in _LABELS:
        raise ParseError("label must be normal, under_extrusion, or unsure")
    try:
        severity = int(data.get("severity"))
    except (TypeError, ValueError) as exc:
        raise ParseError("severity must be an integer from 0 to 3") from exc
    if severity < 0 or severity > 3:
        raise ParseError("severity must be an integer from 0 to 3")
    confidence = data.get("confidence")
    if confidence is None:
        confidence = 0.0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ParseError("confidence must be numeric") from exc
    return {
        "label": label,
        "severity": severity,
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": str(data.get("evidence", "")),
        "abstained": False,
    }


def abstained_result(reason: str) -> dict[str, Any]:
    return {
        "label": None,
        "severity": None,
        "confidence": 0.0,
        "evidence": "",
        "abstained": True,
        "error": reason,
    }


def _load_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise ParseError("model output did not contain JSON") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ParseError("model output JSON must be an object")
    return value
