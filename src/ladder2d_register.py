"""Registration before patch comparison (docs/PROTOCOL_2d_ladder.md, Amendment 2).

Iris: every image to one canonical frame (limbus centre at the image centre, limbus radius 80 pixels) from the existing
integro-differential segmentation (UTIRIS) or the pupil centre (UPOL, Amendment 2a). Fundus: every image to one
disc-fovea similarity frame (Amendment 2b). The same frame is applied to the person, the strangers and the mean
members, so no candidate is favoured by registration. Views expose get(name) / mean(names) like ladder2d.Images,
so the scoring and figure code runs unchanged on registered images.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

import iris_codes as ic
import ladder2d as L

CANON_R = 80.0
IRIS_BOX = (slice(80, 144), slice(136, 200))


def box_resize_rgb(a: np.ndarray, size: int) -> np.ndarray:
    return np.stack([L.box_resize(a[..., c], size) for c in range(a.shape[2])], -1).astype(np.float32)


# ---------- iris ----------

def _warp_to_canvas(luma: np.ndarray, cx: float, cy: float, k: float) -> np.ndarray:
    """Output pixel (u, v) samples input (cx + k (u - c), cy + k (v - c)), c the canvas centre."""
    from skimage.transform import SimilarityTransform, warp
    c = (L.BASE - 1) / 2
    inverse = SimilarityTransform(scale=k, translation=(cx - k * c, cy - k * c))
    return warp(luma, inverse, output_shape=(L.BASE, L.BASE), order=1, mode="constant", cval=0.0).astype(np.float32)


def _luma(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        h = round(im.height * ic.WORK_WIDTH / im.width)
        return np.asarray(im.resize((ic.WORK_WIDTH, h), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0


def pupil_canonical(path: str) -> tuple:
    """UPOL (Amendment 2a): pupil centre to the canvas centre, the working width mapped to the canvas width."""
    try:
        px, py, _ = ic.pupil_blob(ic.load_grey(path))
    except ValueError as e:
        return None, {"ok": False, "error": type(e).__name__}
    return _warp_to_canvas(_luma(path), px, py, ic.WORK_WIDTH / L.BASE), {"ok": True, "error": ""}


def iris_canonical(path: str) -> tuple:
    """(224 x 224 grey canonical image, qc dict). The segmentation runs on iris_codes' working image; the warp samples
    the luma image at the same working width, so circles and pixels share coordinates."""
    work = ic.load_grey(path)
    try:
        s = ic.segment(work)
        ok = 0.15 <= s["ratio"] <= 0.75 and s["offset"] <= 0.25
        qc = {"ratio": s["ratio"], "offset": s["offset"], "ok": bool(ok), "error": ""}
    except ValueError as e:
        return None, {"ok": False, "error": type(e).__name__}
    if not ok:
        return None, qc
    ix, iy, ri = s["iris"]
    return _warp_to_canvas(_luma(path), ix, iy, ri / CANON_R), qc


class IrisView:
    """Canonical images, computed once per image; the fellow eye is flipped in the canonical frame by the caller."""

    def __init__(self, paths: dict, frame: str = "limbus"):
        self.b, self.qc = {}, {}
        canon = iris_canonical if frame == "limbus" else pupil_canonical
        for name, p in paths.items():
            img, qc = canon(p)
            self.qc[name] = qc
            if qc["ok"]:
                self.b[name] = img

    def ok(self, name: str) -> bool:
        return name in self.b

    def get(self, name: str, big: bool = False) -> np.ndarray:
        return self.b[name]

    def mean(self, names: list, big: bool = False) -> np.ndarray:
        return np.mean([self.b[n] for n in names], axis=0)


# ---------- fundus ----------

DISC_R, FOVEA_R = np.array([330.0, 256.0]), np.array([181.0, 256.0])  # disc right of fovea; mirrored otherwise


def fundus_landmarks(rgb512: np.ndarray):
    """(disc xy, fovea xy) on the 512 image, or None (Amendment 2b)."""
    fov = rgb512[..., 0] > 0.1
    if fov.sum() < 1000:
        return None
    r = np.sqrt(fov.sum() / np.pi)
    inner = ndimage.binary_erosion(fov, iterations=25)
    bright = np.where(inner, ndimage.gaussian_filter(rgb512[..., 0] + rgb512[..., 1], 12), -np.inf)
    dy, dx = np.unravel_index(np.argmax(bright), bright.shape)
    yy, xx = np.mgrid[:rgb512.shape[0], :rgb512.shape[1]]
    d = np.hypot(xx - dx, yy - dy)
    band = (d > 0.35 * r) & (d < 0.75 * r) & (np.abs(yy - dy) < 0.25 * r) & inner
    if not band.any():
        return None
    dark = np.where(band, ndimage.gaussian_filter(rgb512[..., 1], 12), np.inf)
    fy, fx = np.unravel_index(np.argmin(dark), dark.shape)
    return np.array([dx, dy], float), np.array([fx, fy], float)


def fundus_canonical(rgb512: np.ndarray):
    from skimage.transform import SimilarityTransform, warp
    lm = fundus_landmarks(rgb512)
    if lm is None:
        return None
    disc, fovea = lm
    targets = (DISC_R, FOVEA_R) if disc[0] > fovea[0] else (FOVEA_R, DISC_R)
    t = SimilarityTransform()
    t.estimate(np.array(targets), np.array([disc, fovea]))  # output -> input
    return warp(rgb512, t, order=1, mode="constant", cval=0.0).astype(np.float32)


class FundusView:
    """Every GRAPE image in the disc-fovea frame, computed once; the caller flips the fellow eye."""

    def __init__(self, imgs: "L.Images"):
        self.b512, self.b, self.qc = {}, {}, {}
        for name in imgs.b512:
            c = fundus_canonical(imgs.get(name, True))
            self.qc[name] = {"ok": c is not None}
            if c is not None:
                self.b512[name] = c
                self.b[name] = box_resize_rgb(c, L.BASE)

    def ok(self, name: str) -> bool:
        return name in self.b

    def get(self, name: str, big: bool = False) -> np.ndarray:
        return self.b512[name] if big else self.b[name]

    def mean(self, names: list, big: bool = False) -> np.ndarray:
        return np.mean([self.get(n, big) for n in names], axis=0)
