"""
Ollama vision validator (yes/no + rationale), ~4k context friendly.
- Saif: uses ollama Python client if available; falls back to HTTP.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import base64

try:
    import ollama  # optional
except Exception:
    ollama = None

import httpx


def build_prompt(
    detections: List[Dict[str, Any]],
    qc_status: str,
    pv_area_sqm_est: Optional[float],
    buffer_radius_sqft: Optional[int]
) -> str:
    """
    Compact prompt: ask for yes/no + rationale, grounded in visual cues.
    Keep it under ~4k tokens by summarizing detections.
    """
    # Summarize top-3 detections by confidence
    det_summary = []
    for d in sorted(detections, key=lambda x: x.get("conf", 0.0), reverse=True)[:3]:
        box = d.get("box", d.get("geometry", []))
        det_summary.append(f"conf={d.get('conf', 0.0):.2f} box={box}")

    lines = [
        "You are a strict rooftop PV validator. Answer exactly with JSON: {\"has_solar\": true|false, \"rationale\": \"text\"}.",
        "Rules:",
        "- Say true only with clear PV cues: rectangular modules, grid-like cells, consistent orientation, specular/glass reflections.",
        "- Avoid false positives: skylights, HVAC, water tanks, roof tiles.",
        "- Consider QC status and buffer. If QC is NOT_VERIFIABLE, be conservative.",
        f"QC status: {qc_status}",
        f"Buffer used (sqft): {buffer_radius_sqft}",
        f"Estimated PV area (sqm, if any): {pv_area_sqm_est}",
        "Detections (top): " + "; ".join(det_summary) if det_summary else "Detections: none",
        "Now respond."
    ]
    return "\n".join(lines)


async def validate_with_ollama(
    ollama_host: str,
    model_name: str,
    image_b64_png: Optional[str],
    detections: List[Dict[str, Any]],
    qc_status: str,
    pv_area_sqm_est: Optional[float],
    buffer_radius_sqft: Optional[int]
) -> Dict[str, Any]:
    """
    Call Ollama vision model (phi3-vision or similar).
    Returns dict: {has_solar: bool, rationale: str} or error.
    """
    prompt = build_prompt(detections, qc_status, pv_area_sqm_est, buffer_radius_sqft)

    # Prefer Python client if available
    if ollama is not None:
        try:
            msgs = [{"role": "user", "content": prompt}]
            if image_b64_png:
                msgs[0]["images"] = [image_b64_png]
            res = ollama.chat(model=model_name, messages=msgs, options={"num_ctx": 4000})
            txt = (res or {}).get("message", {}).get("content", "")
            return _parse_json_response(txt)
        except Exception as e:
            return {"error": f"ollama_client_failed:{e}"}

    # Fallback: HTTP /api/chat
    try:
        payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
        if image_b64_png:
            payload["messages"][0]["images"] = [image_b64_png]
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{ollama_host.rstrip('/')}/api/chat"
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                return {"error": f"ollama_http_error:{r.status_code}"}
            data = r.json()
            txt = (data or {}).get("message", {}).get("content", "")
            return _parse_json_response(txt)
    except Exception as e:
        return {"error": f"ollama_http_exception:{e}"}


def _parse_json_response(txt: str) -> Dict[str, Any]:
    """
    Extract {"has_solar": bool, "rationale": "..."} from model output robustly.
    """
    try:
        import json, re
        # Find a JSON blob in the text
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return {"has_solar": False, "rationale": "No JSON found in LLM output."}
        obj = json.loads(m.group(0))
        return {
            "has_solar": bool(obj.get("has_solar", False)),
            "rationale": str(obj.get("rationale", ""))[:800]  # cap for context
        }
    except Exception:
        return {"has_solar": False, "rationale": "Failed to parse LLM output."}