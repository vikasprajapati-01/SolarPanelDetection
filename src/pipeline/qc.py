"""
QC-lite metrics for quick verifiability checks.
- Saif: Only metrics, no pixel modification.
"""

from __future__ import annotations
from typing import Dict
import cv2
import numpy as np

def qc_blur_variance(gray: np.ndarray) -> float:
    # Variance of Laplacian → lower means blurrier image
    try:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0

def qc_shadow_ratio(gray: np.ndarray) -> float:
    # Estimate shadow as very dark pixels relative to mean
    try:
        thr = max(10, int(gray.mean() * 0.5))
        return float((gray < thr).sum() / gray.size)
    except Exception:
        return 0.0

def qc_glare_ratio(gray: np.ndarray) -> float:
    # Estimate glare as very bright pixels relative to mean
    try:
        thr = min(245, int(200 + (gray.mean() * 0.3)))
        return float((gray > thr).sum() / gray.size)
    except Exception:
        return 0.0

def qc_resolution_ok(h: int, w: int, img_size_target: int = 1280) -> bool:
    # After scaling long side to target, ensure short side is large enough
    try:
        long_side = max(h, w)
        scale = img_size_target / float(long_side) if long_side > 0 else 0.0
        short_scaled = int(min(h, w) * scale)
        return short_scaled >= (6 * 20)  # heuristic: min panel pixel width * factor
    except Exception:
        return False

def qc_lite_from_bgr(bgr: np.ndarray) -> Dict[str, float | bool | str]:
    """
    Compute QC metrics; decide VERIFIABLE/NOT_VERIFIABLE with rationale.
    """
    try:
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        blur = qc_blur_variance(gray)
        shadow = qc_shadow_ratio(gray)
        glare = qc_glare_ratio(gray)
        resolution_ok = qc_resolution_ok(h, w, img_size_target=1280)

        # Thresholds (tunable)
        BLUR_VAR_THRESHOLD = 100.0
        SHADOW_RATIO_THRESHOLD = 0.45
        GLARE_RATIO_THRESHOLD = 0.35

        critical_fail = (blur < BLUR_VAR_THRESHOLD) or \
                        (shadow > SHADOW_RATIO_THRESHOLD) or \
                        (glare > GLARE_RATIO_THRESHOLD) or \
                        (not resolution_ok)

        status = "VERIFIABLE" if not critical_fail else "NOT_VERIFIABLE"
        rationale = []
        if blur < BLUR_VAR_THRESHOLD: rationale.append("blur_variance_low")
        if shadow > SHADOW_RATIO_THRESHOLD: rationale.append("shadow_ratio_high")
        if glare > GLARE_RATIO_THRESHOLD: rationale.append("glare_ratio_high")
        if not resolution_ok: rationale.append("resolution_inadequate")

        return {
            "status": status,
            "blur_variance": blur,
            "shadow_ratio": shadow,
            "glare_ratio": glare,
            "resolution_ok": bool(resolution_ok),
            "rationale": "; ".join(rationale) if rationale else "all_checks_passed",
            "h": h,
            "w": w,
        }
    except Exception:
        return {
            "status": "ERROR",
            "rationale": "qc_failed",
            "blur_variance": 0.0,
            "shadow_ratio": 0.0,
            "glare_ratio": 0.0,
            "resolution_ok": False,
            "h": 0,
            "w": 0,
        }