import cv2
import numpy as np

def preprocess_bgr(bgr: np.ndarray) -> np.ndarray:
    try:
        h, w = bgr.shape[:2]
        # Resize long side to 1280 (downscale only)
        long_side = max(h, w)
        scale = 1280.0 / float(long_side) if long_side > 0 else 1.0
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # CLAHE on L channel
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # Mild denoise
        bgr2 = cv2.fastNlMeansDenoisingColored(bgr2, None, 3, 3, 7, 21)
        return bgr2
    except Exception:
        return bgr