"""Small helpers shared by the window, the PDF report and the referral letter."""
import datetime as dt

import numpy as np
from PIL import Image, ImageOps


def open_xray(path):
    """The X-ray upright (EXIF rotation) with 16-bit / float exports stretched to 8-bit."""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img, np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        img = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8))
    return img


def nice_date(iso):
    try:
        return dt.date.fromisoformat(iso[:10]).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return iso or ""
