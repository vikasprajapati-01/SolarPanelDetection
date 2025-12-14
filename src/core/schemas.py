from __future__ import annotations
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field


class BoxDetection(BaseModel):
    box: List[float] = Field(..., description="xyxy coordinates [x1,y1,x2,y2]")
    conf: float = Field(..., description="confidence score 0..1")


class MaskDetection(BaseModel):
    polygon: List[List[float]] = Field(..., description="list of [x,y] points")
    conf: float = Field(..., description="confidence score 0..1")


class BBoxOrMask(BaseModel):
    detections: List[BoxDetection] = Field(default_factory=list)
    masks: List[MaskDetection] = Field(default_factory=list)


class ModelsUsed(BaseModel):
    detection: Optional[str] = None
    segmentation: Optional[str] = None


class ImageMetadata(BaseModel):
    source: Optional[str] = None
    capture_date: Optional[str] = None
    resolution_estimate: Optional[str] = None
    zoom_parameters: Optional[str] = None
    created_at: Optional[str] = None
    models_used: ModelsUsed = Field(default_factory=ModelsUsed)


class ProviderInfo(BaseModel):
    name: Optional[str] = None
    workflow_id: Optional[str] = None
    fill_ratio: Optional[float] = None


class VerifyUploadResponse(BaseModel):
    sample_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    has_solar: bool
    confidence: Optional[float] = None
    pv_area_sqm_est: Optional[float] = None
    buffer_radius_sqft: Optional[int] = None
    qc_status: Literal["VERIFIABLE", "NOT_VERIFIABLE", "ERROR"]
    bbox_or_mask: BBoxOrMask
    image_metadata: ImageMetadata
    qc_rationale: Optional[str] = None
    area_units: Literal["pixels", "sqm"]
    provider_info: ProviderInfo
    preprocess_applied: Optional[bool] = None

    # Optional media
    overlay_png_base64: Optional[str] = None
    overlay_preprocessed_png_base64: Optional[str] = None
    original_png_base64: Optional[str] = None

    # Optional raw provider blob
    raw: Optional[dict] = None


class ValidatorResult(BaseModel):
    has_solar: bool
    validator_rationale: str


class ValidateUploadResponse(BaseModel):
    has_solar: bool
    validator_rationale: str
    qc_status: Literal["VERIFIABLE", "NOT_VERIFIABLE", "ERROR"]
    buffer_radius_sqft: Optional[int] = None
    pv_area_sqm_est: Optional[float] = None
    area_units: Literal["pixels", "sqm"]
    detections: List[BoxDetection] = Field(default_factory=list)

    # Frontend helpers
    preprocess_applied: Optional[bool] = None
    original_png_base64: Optional[str] = None
    overlay_preprocessed_png_base64: Optional[str] = None
    overlay_png_base64: Optional[str] = None