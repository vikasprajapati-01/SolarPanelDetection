from fastapi import APIRouter, UploadFile, File, Query
from fastapi import Body
import httpx

import base64
import csv
import io
import zipfile
import asyncio
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

from .core.config import RUNTIME_CONFIG
from .pipeline.pipeline import run_pipeline_upload_full
from .pipeline.intelligence.ollama import validate_with_ollama
from .pipeline.sources.static_maps import fetch_google_static_map, fetch_bing_imagery
from .pipeline.validation.response import build_verify_upload_response, build_validate_upload_response
from .pipeline.model.model import ModelProviderFactory  


router = APIRouter()

@router.get("/health")
async def health():
    cfg = RUNTIME_CONFIG.settings
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{cfg.ollama_host}/api/tags")
            ollama_ok = (r.status_code == 200)
    except Exception:
        ollama_ok = False
    return {
        "status": "ok",
        "source_provider": cfg.source_provider,
        "model_provider": cfg.model_provider,
        "validate": cfg.validate,
        "preprocess": cfg.preprocess,
        "ollama_reachable": ollama_ok
    }

@router.get("/config")
async def get_config():
    cfg = RUNTIME_CONFIG.settings
    return {"config": cfg.__dict__}

@router.post("/config")
async def set_config(
    source_provider: str | None = None,
    model_provider: str | None = None,
    validate: str | None = None,
    preprocess: str | None = None,
    det_conf_thr: float | None = None,
    det_iou_thr: float | None = None,
    inference_img_size: int | None = None,
    fill_ratio: float | None = None,
    use_cache: bool | None = None
):
    cfg = RUNTIME_CONFIG.update(
        source_provider=source_provider,
        model_provider=model_provider,
        validate=validate,
        preprocess=preprocess,
        det_conf_thr=det_conf_thr,
        det_iou_thr=det_iou_thr,
        inference_img_size=inference_img_size,
        fill_ratio=fill_ratio,
        use_cache=use_cache
    )
    return {"message": "updated", "config": cfg.__dict__}

@router.post("/model/test")
async def model_test(
    file: UploadFile = File(...),
    include_raw: bool = Query(False, description="Include provider raw output (sanitized)")
):
    """
    Quick provider test endpoint.
    """
    try:
        img_bytes = await file.read()
        cfg = RUNTIME_CONFIG.settings
        provider = ModelProviderFactory.create(cfg)
        res = await provider.infer(
            image_bytes=img_bytes,
            conf=cfg.det_conf_thr,
            iou=cfg.det_iou_thr,
            img_size=cfg.inference_img_size,
            fill_ratio=cfg.fill_ratio,
            use_cache=cfg.use_cache,
        )
        payload = {
            "detections": res.get("detections", []),
            "provider_info": res.get("provider_info", {}),
        }
        if include_raw:
            payload["raw"] = res.get("raw")
        return payload
    except Exception as e:
        return {"error": f"test_failed:{e}"}

@router.post("/verify/upload")
async def verify_upload(
    file: UploadFile = File(...),
    include_overlay: bool = Query(False),
    include_raw: bool = Query(False),
    gsd_m_per_px: float | None = Query(None),
    center_x_px: float | None = Query(None),
    center_y_px: float | None = Query(None),
    preprocess: bool | None = Query(None, description="Override preprocessing: true/false"),  # NEW
):
    img_bytes = await file.read()
    result = await run_pipeline_upload_full(
        image_bytes=img_bytes,
        include_overlay=include_overlay,
        include_raw=include_raw,
        gsd_m_per_px=gsd_m_per_px,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        force_preprocess=preprocess,  # NEW
    )
    return result
    

@router.post("/validate/upload")
async def validate_upload(
    file: UploadFile = File(...),
    gsd_m_per_px: float | None = Query(None),
    center_x_px: float | None = Query(None),
    center_y_px: float | None = Query(None),
    model_name: str = Query("llava-phi3"),
    include_overlay: bool = Query(False),
    preprocess: bool | None = Query(None, description="Override preprocessing: true/false"),  # NEW
):
    img_bytes = await file.read()
    result = await run_pipeline_upload_full(
        image_bytes=img_bytes,
        include_overlay=include_overlay,
        include_raw=False,
        gsd_m_per_px=gsd_m_per_px,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        force_preprocess=preprocess,  # NEW
    )
    if "error" in result:
        return {"error": result["error"]}

    # Choose image for validator: overlay if requested, else original
    image_b64_png = result.get("overlay_png_base64") if include_overlay else result.get("original_png_base64")

    dets = [{"box": d.get("box"), "conf": d.get("conf")} for d in result.get("bbox_or_mask", {}).get("detections", [])]
    cfg = RUNTIME_CONFIG.settings
    val = await validate_with_ollama(
        ollama_host=cfg.ollama_host,
        model_name=model_name,
        image_b64_png=image_b64_png,
        detections=dets,
        qc_status=result.get("qc_status", "VERIFIABLE"),
        pv_area_sqm_est=result.get("pv_area_sqm_est"),
        buffer_radius_sqft=result.get("buffer_radius_sqft"),
    )

    return {
        "has_solar": bool(val.get("has_solar", False)),
        "validator_rationale": val.get("rationale", ""),
        "qc_status": result.get("qc_status"),
        "buffer_radius_sqft": result.get("buffer_radius_sqft"),
        "pv_area_sqm_est": result.get("pv_area_sqm_est"),
        "area_units": result.get("area_units"),
        "detections": dets,
        "preprocess_applied": result.get("preprocess_applied"),
        "original_png_base64": result.get("original_png_base64"),
        "overlay_preprocessed_png_base64": result.get("overlay_preprocessed_png_base64"),
        "overlay_png_base64": result.get("overlay_png_base64") if include_overlay else None,
    }


@router.post("/verify/latlon")
async def verify_latlon(
    lat: float = Query(...),
    lon: float = Query(...),
    provider: str = Query("google", description="google or bing"),
    zoom: int = Query(19),
    width: int = Query(640),
    height: int = Query(640),
    scale: int = Query(2, description="Google only: 1 or 2"),
    include_overlay: bool = Query(True),
    preprocess: bool | None = Query(None),
    gsd_m_per_px: float | None = Query(None, description="Ground sample distance in meters/pixel"),
    center_x_px: float | None = Query(None),
    center_y_px: float | None = Query(None),
):
    """
    Fetch satellite image for lat/lon, run full pipeline, return normalized JSON.
    """
    # Fetch imagery
    if provider.lower() == "google":
        fetched = await fetch_google_static_map(lat, lon, zoom=zoom, size=(width, height), scale=scale, maptype="satellite")
    elif provider.lower() == "bing":
        fetched = await fetch_bing_imagery(lat, lon, zoom=zoom, size=(width, height), maptype="Aerial")
    else:
        return {"error": "unsupported_provider"}

    if "error" in fetched:
        return {"error": fetched["error"]}

    # Run pipeline
    result = await run_pipeline_upload_full(
        image_bytes=fetched["image_bytes"],
        include_overlay=include_overlay,
        include_raw=False,
        gsd_m_per_px=gsd_m_per_px,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        force_preprocess=preprocess,
    )

    # Attach minimal metadata
    result["image_metadata"] = {
        "source": fetched.get("source"),
        "capture_date": None,
        "resolution_estimate": None,
        "zoom_parameters": {"zoom": fetched.get("zoom"), "size": fetched.get("size"), "scale": fetched.get("scale", None)},
        "created_at": None,
        "models_used": {"detection": result.get("provider_info", {}).get("name"), "segmentation": None}
    }

    return build_verify_upload_response(result).dict()


@router.post("/validate/latlon")
async def validate_latlon(
    lat: float = Query(...),
    lon: float = Query(...),
    provider: str = Query("google"),
    zoom: int = Query(19),
    width: int = Query(640),
    height: int = Query(640),
    scale: int = Query(2),
    include_overlay: bool = Query(False),
    preprocess: bool | None = Query(None),
    gsd_m_per_px: float | None = Query(None),
    center_x_px: float | None = Query(None),
    center_y_px: float | None = Query(None),
    model_name: str = Query("llava-phi3"),
):
    """
    Fetch satellite image for lat/lon, run pipeline, then call Ollama validator.
    """
    # Fetch imagery
    if provider.lower() == "google":
        fetched = await fetch_google_static_map(lat, lon, zoom=zoom, size=(width, height), scale=scale, maptype="satellite")
    elif provider.lower() == "bing":
        fetched = await fetch_bing_imagery(lat, lon, zoom=zoom, size=(width, height), maptype="Aerial")
    else:
        return {"error": "unsupported_provider"}

    if "error" in fetched:
        return {"error": fetched["error"]}

    # Run pipeline
    pipeline = await run_pipeline_upload_full(
        image_bytes=fetched["image_bytes"],
        include_overlay=include_overlay,
        include_raw=False,
        gsd_m_per_px=gsd_m_per_px,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        force_preprocess=preprocess,
    )

    if "error" in pipeline:
        return {"error": pipeline["error"]}

    # Choose image for validator
    image_b64_png = pipeline.get("overlay_png_base64") if include_overlay else pipeline.get("original_png_base64")

    dets = [{"box": d.get("box"), "conf": d.get("conf")} for d in pipeline.get("bbox_or_mask", {}).get("detections", [])]
    cfg = RUNTIME_CONFIG.settings
    val = await validate_with_ollama(
        ollama_host=cfg.ollama_host,
        model_name=model_name,
        image_b64_png=image_b64_png,
        detections=dets,
        qc_status=pipeline.get("qc_status", "VERIFIABLE"),
        pv_area_sqm_est=pipeline.get("pv_area_sqm_est"),
        buffer_radius_sqft=pipeline.get("buffer_radius_sqft"),
    )

    return build_validate_upload_response(val, pipeline).dict()


@router.post("/verify/batch")
async def verify_batch_csv(
    file: UploadFile = File(..., description="CSV with columns: sample_id, lat, lon"),
    provider: str = Query("google"),
    zoom: int = Query(19),
    width: int = Query(640),
    height: int = Query(640),
    scale: int = Query(2),
    include_overlay: bool = Query(True),
    preprocess: bool | None = Query(None),
    gsd_m_per_px: float | None = Query(None),
):
    """
    Batch process CSV rows (sample_id, lat, lon).
    Returns aggregated JSON list and a ZIP (base64) of overlays named {sample_id}.png.
    """

    contents = await file.read()
    text = contents.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))

    rows = []
    for row in reader:
        try:
            sample_id = str(row.get("sample_id") or "").strip()
            lat = float(row.get("lat") or row.get("latitude"))
            lon = float(row.get("lon") or row.get("longitude"))
            rows.append({"sample_id": sample_id, "lat": lat, "lon": lon})
        except Exception:
            # skip malformed row
            continue

    async def process_row(r: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[bytes]]:
        # Fetch imagery
        if provider.lower() == "google":
            fetched = await fetch_google_static_map(r["lat"], r["lon"], zoom=zoom, size=(width, height), scale=scale, maptype="satellite")
        else:
            fetched = await fetch_bing_imagery(r["lat"], r["lon"], zoom=zoom, size=(width, height), maptype="Aerial")

        if "error" in fetched:
            return ({"sample_id": r["sample_id"], "lat": r["lat"], "lon": r["lon"], "error": fetched["error"]}, None)

        # Run pipeline
        pipeline = await run_pipeline_upload_full(
            image_bytes=fetched["image_bytes"],
            include_overlay=include_overlay,
            include_raw=False,
            gsd_m_per_px=gsd_m_per_px,
            center_x_px=None,
            center_y_px=None,
            force_preprocess=preprocess,
        )

        # Attach sample_id and coords
        pipeline["sample_id"] = r["sample_id"]
        pipeline["lat"] = r["lat"]
        pipeline["lon"] = r["lon"]
        pipeline["image_metadata"] = {
            "source": fetched.get("source"),
            "capture_date": None,
            "resolution_estimate": None,
            "zoom_parameters": {"zoom": fetched.get("zoom"), "size": fetched.get("size"), "scale": fetched.get("scale", None)},
            "created_at": None,
            "models_used": {"detection": pipeline.get("provider_info", {}).get("name"), "segmentation": None}
        }

        resp = build_verify_upload_response(pipeline).dict()

        # Prepare overlay image bytes for ZIP
        overlay_b64 = resp.get("overlay_png_base64") or resp.get("original_png_base64")
        overlay_bytes: Optional[bytes] = None
        if overlay_b64:
            try:
                overlay_bytes = base64.b64decode(overlay_b64)
            except Exception:
                overlay_bytes = None

        return (resp, overlay_bytes)

    # Run concurrently
    results: List[Tuple[Dict[str, Any], Optional[bytes]]] = await asyncio.gather(
        *[process_row(r) for r in rows], return_exceptions=False
    )

    # Build ZIP of overlays
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for resp, overlay_bytes in results:
            sid = resp.get("sample_id") or "unknown"
            if overlay_bytes:
                zf.writestr(f"{sid}.png", overlay_bytes)
    zip_b64 = base64.b64encode(zip_buf.getvalue()).decode("utf-8")

    aggregated_json = [resp for resp, _ in results]

    return {
        "count": len(aggregated_json),
        "results": aggregated_json,
        "overlays_zip_base64": zip_b64
    }