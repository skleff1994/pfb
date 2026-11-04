"""Offline GelSight reconstruction from recorded raw images.

Uses gs_sdk's own Reconstructor (the same code as the online normalflow node) with the session's
background image and calibration model, so the offline depth is the online one, reproducibly. Needs
torch and OpenCV (pixi environment `hw`) and gs_sdk, found on the path or through
GS_SDK_PATH=<gelsight_ws>/src/gs_sdk; the rest of `hw` does not.

Depth convention: gs_sdk returns a height map H in pixel units with the indentation going negative.
depth_mm = (H_ref - H) * mm_per_px, positive into the gel, with H_ref the median height outside the
contact mask. The config key 'ppmm' of gsmini.yaml is in fact mm per pixel (0.0634 mm/px: 320 px = 20.3 mm).
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_reconstructor():
    try:
        from gs_sdk.gs_reconstruct import Reconstructor
    except ImportError:
        path = os.environ.get("GS_SDK_PATH")
        if not path:
            raise ImportError("gs_sdk not importable; set GS_SDK_PATH=<gelsight_ws>/src/gs_sdk") from None
        sys.path.insert(0, os.path.expanduser(path))
        from gs_sdk.gs_reconstruct import Reconstructor
    return Reconstructor


@dataclass
class Surface:
    depth_mm: np.ndarray   # (H, W) indentation depth, positive into the gel
    mask: np.ndarray       # (H, W) bool contact mask
    normals: np.ndarray    # (H, W, 3) surface normals (image coordinates)
    mm_per_px: float


class OfflineReconstructor:
    def __init__(self, session, model_path=None, device="cpu", contact_mode=None):
        Reconstructor = _import_reconstructor()
        info = session.latched_json("/gelsight/reconstruction_info") or \
            session.manifest.get("reconstruction", {}).get("reconstruction_info") or {}
        if model_path is None:
            local = session.dir / "calib_model.pth"
            model_path = str(local) if local.exists() else info.get("calib_model_path")
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError("calibration model not found; pass model_path")
        self.info = info
        self.mm_per_px = float(info.get("config", {}).get("ppmm", 0.0634))
        self.recon = Reconstructor(model_path, contact_mode=contact_mode or info.get("contact_mode", "standard"),
                                   device=device)
        self.recon.load_bg(session.background())

    def surface(self, image, encoding="rgb8"):
        """Depth, mask and normals of one raw image (as recorded, rgb8 by default)."""
        bgr = image[..., ::-1] if encoding == "rgb8" else image
        G, H, C = self.recon.get_surface_info(np.ascontiguousarray(bgr), self.mm_per_px)
        C = C.astype(bool)
        H_ref = np.median(H[~C]) if (~C).any() else np.median(H)
        depth = (H_ref - H) * self.mm_per_px
        N = np.dstack([-G, np.ones(G.shape[:2] + (1,))])  # as normalflow.utils.gxy2normal
        N /= np.linalg.norm(N, axis=-1, keepdims=True)
        return Surface(depth.astype(np.float32), C, N, self.mm_per_px)
