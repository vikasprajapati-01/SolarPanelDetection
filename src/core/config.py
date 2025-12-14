import os
from typing import Optional
from dataclasses import dataclass

# Runtime-overridable via /config endpoints

@dataclass
class Settings:
    # Providers and toggles
    source_provider: str = os.getenv("SOURCE_PROVIDER", "upload")  # upload | google | bing
    model_provider: str = os.getenv("MODEL_PROVIDER", "roboflow")  # roboflow | local
    validate: str = os.getenv("VALIDATE", "on")                    # on | off
    preprocess: str = os.getenv("PREPROCESS", "on")                # on | off

    # Roboflow
    roboflow_api_key: Optional[str] = os.getenv("ROBOFLOW_API_KEY")
    roboflow_api_url: str = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
    roboflow_workspace: Optional[str] = os.getenv("ROBOFLOW_WORKSPACE")
    roboflow_workflow_id: Optional[str] = os.getenv("ROBOFLOW_WORKFLOW_ID")

    # Imagery sources
    google_static_maps_key: Optional[str] = os.getenv("GOOGLE_STATIC_MAPS_KEY")  # Google Static Maps
    bing_maps_key: Optional[str] = os.getenv("BING_MAPS_KEY")                    # Bing Imagery API

    # Legacy/optional keys (kept for compatibility)
    google_maps_api_key: Optional[str] = os.getenv("GOOGLE_MAPS_API_KEY")
    esri_api_token: Optional[str] = os.getenv("ESRI_API_TOKEN")

    # Validator
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Tuning
    det_conf_thr: float = float(os.getenv("DETECTION_CONF_THR", "0.55"))
    det_iou_thr: float = float(os.getenv("DETECTION_IOU_THR", "0.50"))
    inference_img_size: int = int(os.getenv("INFERENCE_IMG_SIZE", "1280"))
    fill_ratio: float = float(os.getenv("FILL_RATIO", "0.65"))
    use_cache: bool = os.getenv("USE_CACHE", "true").lower() == "true"

    # Local model
    yolo_weights: Optional[str] = os.getenv("YOLO_WEIGHTS")


class RuntimeConfig:
    """
    Holds current settings with runtime-overridable fields (via /config endpoints).
    """
    def __init__(self) -> None:
        self._settings = Settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    def update(self, **kwargs) -> Settings:
        # Simple runtime override (non-persistent).
        for k, v in kwargs.items():
            if hasattr(self._settings, k) and v is not None:
                try:
                    if k in ("det_conf_thr", "det_iou_thr", "fill_ratio"):
                        setattr(self._settings, k, float(v))
                    elif k in ("inference_img_size",):
                        setattr(self._settings, k, int(v))
                    else:
                        setattr(self._settings, k, v)
                except Exception:
                    pass
        return self._settings


# Singleton runtime config instance
RUNTIME_CONFIG = RuntimeConfig()