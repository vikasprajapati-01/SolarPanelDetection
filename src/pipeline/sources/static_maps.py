"""
Static maps fetchers for lat/lon imagery.
Supports Google Static Maps and Bing Imagery API.
Reads keys from env via RUNTIME_CONFIG.
"""

from __future__ import annotations
from typing import Optional, Tuple
import io
import base64
import httpx
import cv2
import numpy as np

from src.core.config import RUNTIME_CONFIG

# Common helper to validate and convert fetched bytes to normalized JPEG bytes
def _to_jpeg_bytes(img_bytes: bytes) -> Optional[bytes]:
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        return bytes(buf.tobytes())
    except Exception:
        return None


async def fetch_google_static_map(
    lat: float,
    lon: float,
    zoom: int = 19,
    size: Tuple[int, int] = (640, 640),
    scale: int = 2,
    maptype: str = "satellite"
) -> dict:
    """
    Fetch a satellite image using Google Static Maps API.
    Requires GOOGLE_STATIC_MAPS_KEY in env.
    """
    key = RUNTIME_CONFIG.settings.google_static_maps_key
    if not key:
        return {"error": "missing_google_static_maps_key"}
    w, h = size
    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}&zoom={zoom}&size={w}x{h}&scale={scale}&maptype={maptype}&key={key}"
    )
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        if r.status_code != 200:
            return {"error": f"google_static_maps_http_{r.status_code}"}
        jpeg = _to_jpeg_bytes(r.content)
        if jpeg is None:
            return {"error": "google_static_maps_decode_failed"}
        return {"image_bytes": jpeg, "source": "google_static_maps", "zoom": zoom, "size": size, "scale": scale}


async def fetch_bing_imagery(
    lat: float,
    lon: float,
    zoom: int = 19,
    size: Tuple[int, int] = (640, 640),
    maptype: str = "Aerial"
) -> dict:
    """
    Fetch a satellite image using Bing Imagery API (Map Tile imagery).
    Requires BING_MAPS_KEY in env.
    """
    key = RUNTIME_CONFIG.settings.bing_maps_key
    if not key:
        return {"error": "missing_bing_maps_key"}
    w, h = size
    # Bing Static Imagery endpoint
    url = (
        "https://dev.virtualearth.net/REST/v1/Imagery/Map/"
        f"{maptype}/{lat},{lon}/{zoom}?mapSize={w},{h}&key={key}"
    )
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        if r.status_code != 200:
            return {"error": f"bing_imagery_http_{r.status_code}"}
        jpeg = _to_jpeg_bytes(r.content)
        if jpeg is None:
            return {"error": "bing_imagery_decode_failed"}
        return {"image_bytes": jpeg, "source": "bing_imagery", "zoom": zoom, "size": size}