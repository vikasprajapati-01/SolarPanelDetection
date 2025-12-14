"""
Unified model provider adapter.
- Saif: call ModelProviderFactory.create(settings).infer(...) from the pipeline.
- It returns normalized detections regardless of provider.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

# Optional imports guarded with try/except so teammates don’t break their env
try:
    from inference_sdk import InferenceHTTPClient  # Roboflow SDK
except Exception:
    InferenceHTTPClient = None

try:
    from ultralytics import YOLO  # Local YOLO (install only when needed)
except Exception:
    YOLO = None

import httpx


class ModelProvider:
    """Interface for detection providers."""
    def load(self) -> None:
        raise NotImplementedError

    async def infer(
        self,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str] = None,
        conf: float = 0.55,
        iou: float = 0.50,
        img_size: int = 1280,
        fill_ratio: float = 0.65,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RoboflowDriver(ModelProvider):
    """
    Roboflow workflow driver.
    - Prefers inference-sdk; falls back to raw httpx if SDK missing.
    - Returns boxes; area estimation happens downstream via fill_ratio.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        workspace: str,
        workflow_id: str,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.client = None  # InferenceHTTPClient or None

    def load(self) -> None:
        # Prepare SDK client if available
        if InferenceHTTPClient is not None:
            try:
                self.client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
            except Exception:
                self.client = None  # Fallback to raw httpx at inference time

    async def infer(
        self,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str] = None,
        conf: float = 0.55,
        iou: float = 0.50,
        img_size: int = 1280,
        fill_ratio: float = 0.65,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        # Saif: we support both bytes and file path; send path to SDK; bytes via base64 to HTTP.
        try:
            detections: List[Dict[str, Any]] = []
            raw_response: Dict[str, Any] = {}

            # 1) Try SDK with image_path (works best for local files)
            if self.client is not None and image_path:
                try:
                    res = self.client.run_workflow(
                        workspace_name=self.workspace,
                        workflow_id=self.workflow_id,
                        images={"image": image_path},
                        use_cache=use_cache,
                    )
                    raw_response = res or {}
                except Exception as e:
                    raw_response = {"sdk_error": str(e)}  # we will fallback to HTTP

            # 2) Fallback to HTTP with base64 or URL
            if not raw_response or ("error" in raw_response):
                payload_image: Dict[str, Any]
                if image_bytes is not None:
                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    payload_image = {"type": "base64", "value": b64}
                elif image_path:
                    # Allows Roboflow to fetch by URL if reachable
                    payload_image = {"type": "url", "value": image_path}
                else:
                    return {
                        "detections": [],
                        "provider_info": {"name": "roboflow", "workflow_id": self.workflow_id},
                        "error": "no_image_supplied",
                    }

                payload = {"api_key": self.api_key, "inputs": {"image": payload_image}}
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        url = f"{self.api_url}/infer/workflows/{self.workspace}/{self.workflow_id}"
                        resp = await client.post(url, headers={"Content-Type": "application/json"}, content=json.dumps(payload))
                        raw_response = resp.json() if resp.status_code == 200 else {"error": f"http_error:{resp.status_code}"}
                except Exception as e:
                    raw_response = {"error": f"http_exception:{e}"}

            # 3) Normalize detections from Roboflow response
            # Roboflow responses can be:
            # - Top-level: {"predictions": [ ... ] }
            # - Workflow: {"outputs": [ { "predictions": { "predictions": [ ... ] } } , ... ] }
            preds: List[Dict[str, Any]] = []

            try:
                # A) Direct top-level predictions list
                if isinstance(raw_response, dict) and isinstance(raw_response.get("predictions"), list):
                    preds = raw_response.get("predictions", [])

                # B) Nested outputs[*].predictions.predictions
                if not preds and isinstance(raw_response.get("outputs"), list):
                    for out in raw_response["outputs"]:
                        try:
                            # Case: predictions is a dict with 'predictions' list
                            if isinstance(out.get("predictions"), dict) and isinstance(out["predictions"].get("predictions"), list):
                                preds.extend(out["predictions"]["predictions"])
                            # Case: predictions is already a list
                            elif isinstance(out.get("predictions"), list):
                                preds.extend(out["predictions"])
                        except Exception:
                            continue
            except Exception:
                preds = []

            # Convert Roboflow center-format boxes to xyxy
            for p in preds:
                try:
                    x = float(p.get("x", 0.0))
                    y = float(p.get("y", 0.0))
                    w = float(p.get("width", 0.0))
                    h = float(p.get("height", 0.0))
                    conf_p = float(p.get("confidence", 0.0))
                    # Center (x,y) → corners
                    x1 = x - w / 2.0
                    y1 = y - h / 2.0
                    x2 = x + w / 2.0
                    y2 = y + h / 2.0
                    detections.append({
                        "type": "box",
                        "geometry": [x1, y1, x2, y2],
                        "confidence": conf_p,
                        "class": p.get("class", "Solarpanel"),
                        "id": p.get("detection_id"),
                    })
                except Exception:
                    continue  # skip malformed entries

            return {
                "detections": detections,
                "provider_info": {
                    "name": "roboflow",
                    "workflow_id": self.workflow_id,
                    "fill_ratio": fill_ratio,  # downstream area estimator for boxes
                },
                "raw": raw_response,  # keep original for audit
            }
        except Exception as e:
            return {
                "detections": [],
                "provider_info": {"name": "roboflow", "workflow_id": self.workflow_id},
                "error": f"provider_failed:{e}",
            }


class LocalYoloDriver(ModelProvider):
    """
    Local YOLO driver (optional).
    - Saif: install ultralytics only when you want to use local .pt.
    - Returns boxes; masks if you load a seg model.
    """

    def __init__(self, weights_path: Optional[str]) -> None:
        self.weights_path = weights_path
        self.model = None

    def load(self) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed. Set MODEL_PROVIDER=roboflow or install ultralytics.")
        if not self.weights_path:
            raise RuntimeError("YOLO weights path not set. Provide YOLO_WEIGHTS in env or runtime config.")
        self.model = YOLO(self.weights_path)

    async def infer(
        self,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str] = None,
        conf: float = 0.55,
        iou: float = 0.50,
        img_size: int = 1280,
        fill_ratio: float = 0.65,
        use_cache: bool = True,  # unused here
    ) -> Dict[str, Any]:
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load() first.")

            # Save bytes to a temp file if needed
            tmp_path = image_path
            if image_bytes is not None and image_path is None:
                import tempfile, os
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                with open(fd, "wb") as f:
                    f.write(image_bytes)
                os.close(fd)

            # Run prediction
            res = self.model.predict(tmp_path, conf=conf, iou=iou, imgsz=img_size, verbose=False)
            detections: List[Dict[str, Any]] = []
            masks: List[Dict[str, Any]] = []
            for r in res:
                # Boxes
                if getattr(r, "boxes", None) is not None:
                    for b, c, s in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        try:
                            xyxy = b.cpu().numpy().tolist()
                            detections.append({
                                "type": "box",
                                "geometry": xyxy,
                                "confidence": float(s.item()),
                                "class": "Solarpanel",
                            })
                        except Exception:
                            continue
                # Masks (seg models only)
                if getattr(r, "masks", None) is not None and r.masks is not None:
                    for m, s in zip(r.masks.xy, r.boxes.conf):
                        try:
                            poly = [[float(x), float(y)] for x, y in m]
                            masks.append({
                                "type": "mask",
                                "geometry": poly,
                                "confidence": float(s.item()),
                                "class": "Solarpanel",
                            })
                        except Exception:
                            continue

            return {
                "detections": detections + masks,  # downstream can separate by type
                "provider_info": {"name": "local-yolo", "weights": self.weights_path},
                "raw": None,
            }
        except Exception as e:
            return {
                "detections": [],
                "provider_info": {"name": "local-yolo", "weights": self.weights_path},
                "error": f"provider_failed:{e}",
            }


class ModelProviderFactory:
    """Returns a provider instance based on settings.model_provider."""
    @staticmethod
    def create(settings) -> ModelProvider:
        name = (settings.model_provider or "roboflow").lower()
        if name == "roboflow":
            drv = RoboflowDriver(
                api_url=settings.roboflow_api_url,
                api_key=settings.roboflow_api_key or "",
                workspace=settings.roboflow_workspace or "",
                workflow_id=settings.roboflow_workflow_id or "",
            )
            drv.load()
            return drv
        elif name == "local":
            drv = LocalYoloDriver(weights_path=settings.yolo_weights)
            drv.load()
            return drv
        else:
            raise ValueError(f"Unknown MODEL_PROVIDER: {settings.model_provider}")