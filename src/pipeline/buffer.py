"""
Buffer selection logic (1200 → 2400 sq ft) with shapely.
- Saif: Requires GSD to convert physical area to pixel radius.
- Circle center defaults to image center if not provided.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
from shapely.geometry import Polygon, Point


SQFT_TO_SQM = 0.092903  # conversion factor


def sqft_to_radius_px(sqft: float, gsd_m_per_px: float) -> float:
    """
    Convert buffer area in sqft to pixel radius using GSD.
    area_m2 = sqft * 0.092903
    circle_area_px = area_m2 / (gsd^2)
    r_px = sqrt(circle_area_px / pi)
    """
    try:
        area_m2 = float(sqft) * SQFT_TO_SQM
        circle_area_px = area_m2 / (gsd_m_per_px ** 2)
        r_px = math.sqrt(circle_area_px / math.pi)
        return float(r_px)
    except Exception:
        return 0.0


def rect_from_xyxy(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    # Build shapely rectangle polygon from corners
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def circle_polygon(cx: float, cy: float, r_px: float, resolution: int = 64) -> Polygon:
    # Approximate circle as shapely buffer of a Point
    return Point(cx, cy).buffer(r_px, resolution=resolution)


def choose_detection_for_buffer(
    detections: List[Dict],
    cx: float,
    cy: float,
    r_px: float
) -> Tuple[Optional[Dict], float]:
    """
    Pick the single detection with largest overlap with the buffer circle.
    Returns: (chosen_detection, overlap_area_px)
    """
    try:
        if r_px <= 0.0:
            return None, 0.0
        circle = circle_polygon(cx, cy, r_px)
        best = None
        best_overlap = 0.0
        for d in detections:
            if d.get("type") != "box":
                continue
            try:
                x1, y1, x2, y2 = d.get("geometry", [0, 0, 0, 0])
                rect = rect_from_xyxy(x1, y1, x2, y2)
                inter = rect.intersection(circle)
                a = float(inter.area) if not inter.is_empty else 0.0
                if a > best_overlap:
                    best = d
                    best_overlap = a
            except Exception:
                continue
        return best, best_overlap
    except Exception:
        return None, 0.0