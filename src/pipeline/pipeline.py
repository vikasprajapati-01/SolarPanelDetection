"""
Upload-mode pipeline (full).
- Detect → QC-lite → Buffer selection → Area conversion → Overlays
- Shows overlay of detections and an overlay of the preprocessed image.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import base64
import cv2
import numpy as np

from .model.model import ModelProviderFactory
from .qc import qc_lite_from_bgr
from .buffer import sqft_to_radius_px, choose_detection_for_buffer
from .preprocess import preprocess_bgr
from ..core.config import RUNTIME_CONFIG


def _draw_overlay(bgr: np.ndarray, detections: List[Dict[str, Any]], cx: float, cy: float, r_px: float, chosen: Optional[Dict[str, Any]], header_text: str) -> np.ndarray:
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = bgr.copy()

    try:
        cv2.circle(rgb, (int(cx), int(cy)), int(r_px), (255, 215, 0), 2)
    except Exception:
        pass

    for d in detections:
        if d.get("type") != "box":
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in d.get("geometry", [0, 0, 0, 0])]
            color = (0, 255, 0)
            thick = 2
            if chosen is not None and d is chosen:
                color = (255, 0, 0)
                thick = 3
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, thick)
            cv2.putText(rgb, f"{d.get('confidence', 0.0):.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception:
            continue

    try:
        cv2.putText(rgb, header_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2)
    except Exception:
        pass

    return rgb


def _encode_png_base64(img_bgr_or_rgb: np.ndarray, is_rgb: bool = True) -> str:
    try:
        bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR) if is_rgb else img_bgr_or_rgb
        ok, buf = cv2.imencode(".png", bgr)
        if ok:
            return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        pass
    return ""

async def run_pipeline_upload_full(
    image_bytes: bytes,
    include_overlay: bool = False,
    include_raw: bool = False,
    gsd_m_per_px: Optional[float] = None,
    center_x_px: Optional[float] = None,
    center_y_px: Optional[float] = None,
    force_preprocess: Optional[bool] = None,  # NEW: frontend toggle
) -> Dict[str, Any]:
    cfg = RUNTIME_CONFIG.settings

    # Decode
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "invalid_image"}
    except Exception:
        return {"error": "decode_failed"}

    h, w = bgr.shape[:2]

    # Decide preprocessing: frontend toggle > env
    use_pre = force_preprocess if force_preprocess is not None else ((cfg.preprocess or "on") == "on")
    bgr_prep = preprocess_bgr(bgr.copy()) if use_pre else bgr.copy()

    # QC-lite
    qc = qc_lite_from_bgr(bgr_prep)

    # Provider inference on selected image
    try:
        provider = ModelProviderFactory.create(cfg)
    except Exception as e:
        return {"error": f"provider_init_failed:{e}", "qc_status": qc.get("status")}

    try:
        ok, buf = cv2.imencode(".jpg", bgr_prep)
        img_bytes_for_model = buf.tobytes() if ok else image_bytes

        res = await provider.infer(
            image_bytes=img_bytes_for_model,
            conf=cfg.det_conf_thr,
            iou=cfg.det_iou_thr,
            img_size=cfg.inference_img_size,
            fill_ratio=cfg.fill_ratio,
            use_cache=cfg.use_cache,
        )
        detections: List[Dict[str, Any]] = res.get("detections", [])
        provider_info = res.get("provider_info", {})
        raw = res.get("raw") if include_raw else None
    except Exception as e:
        return {"error": f"infer_failed:{e}", "qc_status": qc.get("status")}

    # Buffer selection
    buffer_radius_sqft_used: Optional[int] = None
    chosen: Optional[Dict[str, Any]] = None
    overlap_px: float = 0.0
    cx = float(center_x_px) if center_x_px is not None else float(w // 2)
    cy = float(center_y_px) if center_y_px is not None else float(h // 2)
    r_px = 0.0

    if gsd_m_per_px is not None:
        try:
            r_px_1200 = sqft_to_radius_px(1200.0, float(gsd_m_per_px))
            cand_1200, ov_1200 = choose_detection_for_buffer(detections, cx, cy, r_px_1200)
            if cand_1200 is not None and ov_1200 > 0.0:
                chosen, overlap_px, buffer_radius_sqft_used, r_px = cand_1200, ov_1200, 1200, r_px_1200
            else:
                r_px_2400 = sqft_to_radius_px(2400.0, float(gsd_m_per_px))
                cand_2400, ov_2400 = choose_detection_for_buffer(detections, cx, cy, r_px_2400)
                if cand_2400 is not None and ov_2400 > 0.0:
                    chosen, overlap_px, buffer_radius_sqft_used, r_px = cand_2400, ov_2400, 2400, r_px_2400
                else:
                    chosen, overlap_px, buffer_radius_sqft_used, r_px = None, 0.0, 2400, r_px_2400
        except Exception:
            chosen, overlap_px, buffer_radius_sqft_used, r_px = None, 0.0, None, 0.0

    # Area estimation
    area_units = "pixels"
    pv_area_sqm_est: Optional[float] = None
    pixel_area_total = 0

    try:
        masks = [d for d in detections if d.get("type") == "mask"]
        boxes = [d for d in detections if d.get("type") == "box"]
        if masks:
            for m in masks:
                pts = np.array(m.get("geometry", []), dtype=np.float32)
                if pts.ndim == 2 and len(pts) >= 3:
                    pixel_area_total += int(abs(cv2.contourArea(pts)))
        elif boxes:
            for b in boxes:
                x1, y1, x2, y2 = b.get("geometry", [0, 0, 0, 0])
                bw = max(0, int(x2 - x1))
                bh = max(0, int(y2 - y1))
                pixel_area_total += bw * bh

        if gsd_m_per_px is not None:
            area_units = "sqm"
            if chosen and chosen.get("type") == "box":
                x1, y1, x2, y2 = chosen.get("geometry", [0, 0, 0, 0])
                bw = max(0, int(x2 - x1))
                bh = max(0, int(y2 - y1))
                fill_ratio = float(provider_info.get("fill_ratio", cfg.fill_ratio))
                px_area = int(bw * bh * fill_ratio)
                pv_area_sqm_est = float(px_area) * (float(gsd_m_per_px) ** 2)
            elif masks:
                pv_area_sqm_est = float(pixel_area_total) * (float(gsd_m_per_px) ** 2)
            else:
                pv_area_sqm_est = None
        else:
            pv_area_sqm_est = None
            area_units = "pixels"
    except Exception:
        pv_area_sqm_est = None
        area_units = "pixels"

    # Decision
    has_solar = bool(detections)
    if gsd_m_per_px is not None:
        has_solar = chosen is not None and overlap_px > 0.0

    # Overlays: return both original and preprocessed previews
    overlay_b64 = ""
    overlay_prep_b64 = ""
    original_png_b64 = ""
    if include_overlay:
        try:
            # Original
            ok0, buf0 = cv2.imencode(".png", bgr)
            if ok0:
                original_png_b64 = base64.b64encode(buf0.tobytes()).decode("utf-8")
            # Preprocessed preview (no drawings)
            okp, bufp = cv2.imencode(".png", bgr_prep)
            if okp:
                overlay_prep_b64 = base64.b64encode(bufp.tobytes()).decode("utf-8")
            # Annotated overlay (on the image used for detection)
            header = f"has_solar={has_solar} area_units={area_units} preprocess={use_pre}"
            overlay_rgb = _draw_overlay(bgr_prep, detections, cx, cy, r_px, chosen, header)
            overlay_b64 = _encode_png_base64(overlay_rgb, is_rgb=True)
        except Exception:
            overlay_b64 = ""
            overlay_prep_b64 = ""
            original_png_b64 = ""

    payload: Dict[str, Any] = {
        "sample_id": None,
        "lat": None,
        "lon": None,
        "has_solar": has_solar,
        "confidence": None,
        "pv_area_sqm_est": pv_area_sqm_est,
        "buffer_radius_sqft": buffer_radius_sqft_used,
        "qc_status": qc.get("status"),
        "bbox_or_mask": {
            "detections": [{"box": d["geometry"], "conf": d.get("confidence", 0.0)} for d in detections if d.get("type") == "box"],
            "masks": [{"polygon": d["geometry"], "conf": d.get("confidence", 0.0)} for d in detections if d.get("type") == "mask"],
        },
        "image_metadata": {
            "source": "upload",
            "capture_date": None,
            "resolution_estimate": None,
            "zoom_parameters": None,
            "created_at": None,
            "models_used": {"detection": provider_info.get("name"), "segmentation": None}
        },
        "qc_rationale": qc.get("rationale"),
        "area_units": area_units,
        "provider_info": provider_info,
        "preprocess_applied": use_pre,  # NEW
    }

    if include_overlay:
        payload["overlay_png_base64"] = overlay_b64 or None
        payload["overlay_preprocessed_png_base64"] = overlay_prep_b64 or None
        payload["original_png_base64"] = original_png_b64 or None  # NEW

    if include_raw and raw is not None:
        payload["raw"] = raw

    return payload

async def run_pipeline_upload(image_bytes: bytes, include_raw: bool = False) -> dict:
    """
    Minimal upload-mode pipeline (used by /model/test).
    - Detect using current provider.
    - Return normalized detections + provider info.
    """
    from .model.model import ModelProviderFactory
    from ..core.config import RUNTIME_CONFIG

    cfg = RUNTIME_CONFIG.settings
    try:
        provider = ModelProviderFactory.create(cfg)
        res = await provider.infer(
            image_bytes=image_bytes,
            conf=cfg.det_conf_thr,
            iou=cfg.det_iou_thr,
            img_size=cfg.inference_img_size,
            fill_ratio=cfg.fill_ratio,
            use_cache=cfg.use_cache,
        )
        payload = {
            "status": "ok",
            "detections": res.get("detections", []),
            "provider_info": res.get("provider_info", {}),
        }
        if include_raw:
            payload["raw"] = res.get("raw")
        return payload
    except Exception as e:
        return {"error": f"infer_failed:{e}"}