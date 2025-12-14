from __future__ import annotations
from typing import List, Dict, Any, Optional

from ..qc import qc_lite_from_bgr
from ...core.schemas import (
    BoxDetection,
    MaskDetection,
    BBoxOrMask,
    ImageMetadata,
    ModelsUsed,
    ProviderInfo,
    VerifyUploadResponse,
    ValidateUploadResponse,
)


def build_verify_upload_response(payload: Dict[str, Any]) -> VerifyUploadResponse:
    """
    Normalize the dict payload from run_pipeline_upload_full into a validated schema.
    """
    boxes = [
        BoxDetection(box=det.get("box"), conf=float(det.get("conf", 0.0)))
        for det in payload.get("bbox_or_mask", {}).get("detections", [])
        if isinstance(det.get("box"), (list, tuple))
    ]
    masks = [
        MaskDetection(polygon=mask.get("polygon"), conf=float(mask.get("conf", 0.0)))
        for mask in payload.get("bbox_or_mask", {}).get("masks", [])
        if isinstance(mask.get("polygon"), (list, tuple))
    ]

    meta = ImageMetadata(
        source=(payload.get("image_metadata") or {}).get("source"),
        capture_date=(payload.get("image_metadata") or {}).get("capture_date"),
        resolution_estimate=(payload.get("image_metadata") or {}).get("resolution_estimate"),
        zoom_parameters=(payload.get("image_metadata") or {}).get("zoom_parameters"),
        created_at=(payload.get("image_metadata") or {}).get("created_at"),
        models_used=ModelsUsed(
            detection=(payload.get("image_metadata") or {}).get("models_used", {}).get("detection"),
            segmentation=(payload.get("image_metadata") or {}).get("models_used", {}).get("segmentation"),
        ),
    )

    provider = ProviderInfo(
        name=(payload.get("provider_info") or {}).get("name"),
        workflow_id=(payload.get("provider_info") or {}).get("workflow_id"),
        fill_ratio=(payload.get("provider_info") or {}).get("fill_ratio"),
    )

    resp = VerifyUploadResponse(
        sample_id=payload.get("sample_id"),
        lat=payload.get("lat"),
        lon=payload.get("lon"),
        has_solar=bool(payload.get("has_solar", False)),
        confidence=payload.get("confidence"),
        pv_area_sqm_est=payload.get("pv_area_sqm_est"),
        buffer_radius_sqft=payload.get("buffer_radius_sqft"),
        qc_status=payload.get("qc_status", "VERIFIABLE"),
        bbox_or_mask=BBoxOrMask(detections=boxes, masks=masks),
        image_metadata=meta,
        qc_rationale=payload.get("qc_rationale"),
        area_units=payload.get("area_units", "pixels"),
        provider_info=provider,
        preprocess_applied=payload.get("preprocess_applied"),
        overlay_png_base64=payload.get("overlay_png_base64"),
        overlay_preprocessed_png_base64=payload.get("overlay_preprocessed_png_base64"),
        original_png_base64=payload.get("original_png_base64"),
        raw=payload.get("raw"),
    )
    return resp


def build_validate_upload_response(
    validator: Dict[str, Any],
    pipeline: Dict[str, Any]
) -> ValidateUploadResponse:
    """
    Combine validator output and pipeline context into a validated schema.
    """
    boxes = [
        BoxDetection(box=det.get("box"), conf=float(det.get("conf", 0.0)))
        for det in pipeline.get("bbox_or_mask", {}).get("detections", [])
        if isinstance(det.get("box"), (list, tuple))
    ]

    resp = ValidateUploadResponse(
        has_solar=bool(validator.get("has_solar", False)),
        validator_rationale=str(validator.get("rationale", "")),
        qc_status=pipeline.get("qc_status", "VERIFIABLE"),
        buffer_radius_sqft=pipeline.get("buffer_radius_sqft"),
        pv_area_sqm_est=pipeline.get("pv_area_sqm_est"),
        area_units=pipeline.get("area_units", "pixels"),
        detections=boxes,
        preprocess_applied=pipeline.get("preprocess_applied"),
        original_png_base64=pipeline.get("original_png_base64"),
        overlay_preprocessed_png_base64=pipeline.get("overlay_preprocessed_png_base64"),
        overlay_png_base64=pipeline.get("overlay_png_base64"),
    )
    return resp