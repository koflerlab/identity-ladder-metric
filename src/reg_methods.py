"""Registration methods of docs/PROTOCOL_registration.md (one function per method code).

Every method returns a Result: the output image, a mapping from output-grid points to raw-image coordinates, a valid
mask and a failure reason ("" when it succeeded). A failure is never replaced by the unregistered image. Points are
(x, y) in pixels with pixel centres at integers. Raw images: GRAPE = the 512 RGB base image in [0, 1]; iris = the
working image of iris_codes (width 320), as a pair of arrays (median-filtered grey 0-255 for segmentation, luma 0-1
for sampling).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

import iris_codes as ic
import ladder2d as L
import ladder2d_register as R

cv2.setNumThreads(1)
SEED = 20261101
BASE = L.BASE  # 224
G_SIZE = 512
IRIS_CANON_R = R.CANON_R  # 80
POLAR = (64, 256)  # rubber sheet: radial x angular samples
ECC_ITER, ECC_EPS, ECC_MAX_SHIFT, ECC_MAX_ROT = 100, 1e-5, 10.0, 5.0
ORB_N, ORB_RATIO, ORB_RANSAC, ORB_ITERS, ORB_MIN_INLIERS = 800, 0.8, 3.0, 500, 12
I4_SHIFT = 16
FLOW_SIGMA = 8.0


# ---------- mappings: output-grid points -> raw coordinates ----------

class Affine:
    def __init__(self, M: np.ndarray):
        self.M = np.asarray(M, float).reshape(3, 3) if np.size(M) == 9 else np.vstack([np.asarray(M, float), [0, 0, 1]])

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float)
        return pts @ self.M[:2, :2].T + self.M[:2, 2]

    def inverse(self) -> "Affine":
        return Affine(np.linalg.inv(self.M))

    def then(self, outer: "Affine") -> "Affine":
        """self maps out -> mid, outer maps mid -> raw; returns out -> raw."""
        return Affine(outer.M @ self.M)


class Chain:
    """inner maps out -> mid (any callable), outer maps mid -> raw."""

    def __init__(self, inner, outer):
        self.inner, self.outer = inner, outer

    def __call__(self, pts):
        return self.outer(self.inner(pts))


class Flow:
    """p -> p + u(p), u sampled bilinearly from a dense (H, W, 2) field on the output grid."""

    def __init__(self, u: np.ndarray):
        self.u = u

    def __call__(self, pts):
        pts = np.asarray(pts, float)
        c = [pts[:, 1], pts[:, 0]]
        du = np.stack([ndimage.map_coordinates(self.u[..., k], c, order=1, mode="nearest") for k in (0, 1)], 1)
        return pts + du


class Shift:
    """Circular shift along x of a periodic angular axis of length n."""

    def __init__(self, s: int, n: int):
        self.s, self.n = s, n

    def __call__(self, pts):
        pts = np.array(pts, float)
        pts[:, 0] = (pts[:, 0] + self.s) % self.n
        return pts


class RubberSheet:
    """(angle column, radius row) -> raw point, linear between pupil and limbus boundary points (iris_codes.normalise
    geometry, POLAR samples)."""

    def __init__(self, pupil, iris, shape=POLAR):
        self.p, self.i, self.shape = pupil, iris, shape

    def __call__(self, pts):
        pts = np.asarray(pts, float)
        (px, py, rp), (ix, iy, ri) = self.p, self.i
        th = 2 * np.pi * pts[:, 0] / self.shape[1]
        t = 0.08 + 0.84 * pts[:, 1] / (self.shape[0] - 1)
        bp = np.stack([px + rp * np.cos(th), py - rp * np.sin(th)], 1)
        bi = np.stack([ix + ri * np.cos(th), iy - ri * np.sin(th)], 1)
        return bp + t[:, None] * (bi - bp)


def grid(shape) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return np.stack([xx.ravel(), yy.ravel()], 1).astype(float)


def remap(raw: np.ndarray, mapping, shape, nearest: bool = False) -> np.ndarray:
    pts = mapping(grid(shape)).astype(np.float32)
    mx, my = pts[:, 0].reshape(shape), pts[:, 1].reshape(shape)
    flag = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(np.ascontiguousarray(raw, dtype=np.float32), mx, my, flag, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)


def similarity_from_two(out_pts, raw_pts) -> Affine:
    """Exact similarity sending two output points to two raw points (complex form z_raw = a z_out + b)."""
    o = np.asarray(out_pts, float) @ [1, 1j]
    r = np.asarray(raw_pts, float) @ [1, 1j]
    a = (r[1] - r[0]) / (o[1] - o[0])
    b = r[0] - a * o[0]
    return Affine([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag], [0, 0, 1]])


def scale_rotation(M) -> tuple:
    M = np.asarray(M, float)
    return float(np.sqrt(abs(np.linalg.det(M[:2, :2])))), float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))


@dataclass
class Result:
    out: np.ndarray | None
    phi: object | None
    valid: np.ndarray | None
    fail: str = ""
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.fail


def failed(reason: str) -> Result:
    return Result(None, None, None, reason)


# ---------- GRAPE ----------

def fov(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] > 0.1


def g0(raw: np.ndarray) -> Result:
    return Result(raw.astype(np.float32), Affine(np.eye(3)), fov(raw))


def g1(raw: np.ndarray) -> Result:
    lm = R.fundus_landmarks(raw)
    if lm is None:
        return failed("no disc-fovea landmark")
    disc, fovea = lm
    targets = (R.DISC_R, R.FOVEA_R) if disc[0] > fovea[0] else (R.FOVEA_R, R.DISC_R)
    phi = similarity_from_two(targets, [disc, fovea])
    out = remap(raw, phi, (G_SIZE, G_SIZE))
    return Result(out, phi, fov(out))


def _clahe_green_u8(rgb: np.ndarray) -> np.ndarray:
    from skimage.exposure import equalize_adapthist
    return (equalize_adapthist(np.clip(rgb[..., 1], 0, 1)) * 255).astype(np.uint8)


def _orb(rgb: np.ndarray):
    mask = ndimage.binary_erosion(fov(rgb), iterations=15).astype(np.uint8)
    return cv2.ORB_create(ORB_N).detectAndCompute(_clahe_green_u8(rgb), mask)


def g2(anchor: Result, moving_raw: np.ndarray, anchor_feats=None) -> Result:
    """ORB + RANSAC similarity of the moving raw image onto the anchor's G0 output (Amendment 2a parameters)."""
    ka, da = anchor_feats if anchor_feats is not None else _orb(anchor.out)
    kb, db = _orb(moving_raw)
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return failed("too few keypoints")
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
            if len(m) == 2 and m[0].distance < ORB_RATIO * m[1].distance]
    if len(good) < ORB_MIN_INLIERS:
        return failed("too few matches")
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(SEED)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ORB_RANSAC,
                                         maxIters=ORB_ITERS)
    if M is None:
        return failed("no RANSAC model")
    s, rot = scale_rotation(M)
    if int(inl.sum()) < ORB_MIN_INLIERS or not (0.8 <= s <= 1.25) or abs(rot) >= 20:
        return failed("RANSAC model rejected")
    phi = Affine(M)
    out = remap(moving_raw, phi, (G_SIZE, G_SIZE))
    return Result(out, phi, fov(out), info={"inliers": int(inl.sum())})


def ecc(anchor: Result, moving: Result, size: int) -> Result:
    """ECC (Euclidean) of moving's output onto anchor's output; the update maps anchor grid -> moving grid."""
    ga, gb = (np.ascontiguousarray(L.grey(r.out), dtype=np.float32) * 255 for r in (anchor, moving))
    mask = (anchor.valid & moving.valid).astype(np.uint8)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(ga, gb, W, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITER, ECC_EPS), mask, 5)
    except cv2.error:
        return failed("ECC did not converge")
    upd = Affine(W)
    c = np.array([[(size - 1) / 2, (size - 1) / 2]])
    if np.linalg.norm(upd(c) - c) > ECC_MAX_SHIFT or abs(scale_rotation(upd.M)[1]) > ECC_MAX_ROT:
        return failed("ECC update too large")
    phi = upd.then(moving.phi) if isinstance(moving.phi, Affine) else Chain(upd, moving.phi)
    return Result(None, phi, None, info={"update": W.tolist()})


def g3(anchor_g1: Result, moving_g1: Result, moving_raw: np.ndarray) -> Result:
    r = ecc(anchor_g1, moving_g1, G_SIZE)
    if not r.ok:
        return r
    r.out = remap(moving_raw, r.phi, (G_SIZE, G_SIZE))
    r.valid = fov(r.out)
    return r


def jacobian_det(u: np.ndarray) -> np.ndarray:
    dux_dy, dux_dx = np.gradient(u[..., 0])
    duy_dy, duy_dx = np.gradient(u[..., 1])
    return (1 + dux_dx) * (1 + duy_dy) - dux_dy * duy_dx


def g4(anchor_g1: Result, g3r: Result, moving_raw: np.ndarray) -> Result:
    """G3, then smoothed DIS flow from the G3 output to the anchor output; folding is a failure."""
    if not g3r.ok:
        return failed("G3 failed")
    u8 = lambda r: (np.clip(L.grey(r.out), 0, 1) * 255).astype(np.uint8)  # noqa: E731
    u = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(u8(anchor_g1), u8(g3r), None)
    u = np.stack([ndimage.gaussian_filter(u[..., k], FLOW_SIGMA) for k in (0, 1)], -1)
    J = jacobian_det(u)[anchor_g1.valid & g3r.valid]
    info = {"jac_nonpos": float((J <= 0).mean()) if J.size else np.nan,
            "jac_p99_abslog": float(np.percentile(np.abs(np.log(np.clip(J, 1e-6, None))), 99)) if J.size else np.nan}
    if J.size == 0 or (J <= 0).any():
        return Result(None, None, None, "folding", info)
    phi = Chain(Flow(u), g3r.phi)
    out = remap(moving_raw, phi, (G_SIZE, G_SIZE))
    return Result(out, phi, fov(out), info=info)


def quantile_reference(values: np.ndarray, n: int = 1001) -> np.ndarray:
    return np.quantile(values, np.linspace(0, 1, n))


def match_to_reference(v: np.ndarray, ref_q: np.ndarray) -> np.ndarray:
    """Histogram matching: each value goes to the reference quantile at its own rank."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    ranks[order] = np.linspace(0, 1, len(v)) if len(v) > 1 else 0.5
    return np.interp(ranks, np.linspace(0, 1, len(ref_q)), ref_q)


def h1_fundus(g1r: Result, ref_q: list) -> Result:
    if not g1r.ok:
        return g1r
    out = g1r.out.copy()
    for c in range(3):
        out[..., c][g1r.valid] = match_to_reference(out[..., c][g1r.valid], ref_q[c])
    return Result(out, g1r.phi, g1r.valid)


def h2_fundus(g1r: Result, cam: dict) -> Result:
    """Per-camera standardisation of each channel inside the field of view (cam: {'mean': [3], 'sd': [3]})."""
    if not g1r.ok:
        return g1r
    out = np.zeros_like(g1r.out)
    for c in range(3):
        out[..., c][g1r.valid] = (g1r.out[..., c][g1r.valid] - cam["mean"][c]) / cam["sd"][c]
    return Result(out, g1r.phi, g1r.valid)


# ---------- iris ----------

@dataclass
class IrisRaw:
    grey: np.ndarray  # median-filtered working image, 0-255 (segmentation)
    luma: np.ndarray  # working luma, 0-1 (sampling)
    orig: tuple       # original (width, height)


def iris_raw(path: str, flip: bool = False) -> IrisRaw:
    from PIL import Image
    g, lu = ic.load_grey(path), R._luma(path)
    with Image.open(path) as im:
        size = im.size
    if flip:
        g, lu = g[:, ::-1].copy(), lu[:, ::-1].copy()
    return IrisRaw(g, lu, size)


def _canvas_valid(raw: IrisRaw, phi, shape) -> np.ndarray:
    return remap(np.ones_like(raw.luma), phi, shape, nearest=True) > 0.5


def _iris_result(raw: IrisRaw, phi, shape=(BASE, BASE), info=None) -> Result:
    return Result(remap(raw.luma, phi, shape), phi, _canvas_valid(raw, phi, shape), info=info or {})


def i0(raw: IrisRaw) -> Result:
    """Centre square crop of the original resized to 224, expressed in working coordinates."""
    w, h = raw.orig
    s = min(w, h)
    k_orig = s / BASE                      # base px -> original px
    k_work = ic.WORK_WIDTH / w             # original px -> working px
    x0, y0 = (w - s) / 2, (h - s) / 2
    # x_work = ((x0 + (u + 0.5) k_orig - 0.5) + 0.5) k_work - 0.5
    a = k_orig * k_work
    bx, by = (x0 + 0.5 * k_orig) * k_work - 0.5, (y0 + 0.5 * k_orig) * k_work - 0.5
    return _iris_result(raw, Affine([[a, 0, bx], [0, a, by], [0, 0, 1]]))


def _canvas_map(cx: float, cy: float, k: float) -> Affine:
    c = (BASE - 1) / 2
    return Affine([[k, 0, cx - k * c], [0, k, cy - k * c], [0, 0, 1]])


def segment(raw: IrisRaw):
    """(segmentation, failure reason) with the 2D Amendment 2 checks."""
    try:
        s = ic.segment(raw.grey)
    except ValueError as e:
        return None, f"segmentation: {e}"
    if not (0.15 <= s["ratio"] <= 0.75 and s["offset"] <= 0.25):
        return None, "segmentation check"
    return s, ""


def i1(raw: IrisRaw) -> Result:
    s, why = segment(raw)
    if s is None:
        return failed(why)
    ix, iy, ri = s["iris"]
    return _iris_result(raw, _canvas_map(ix, iy, ri / IRIS_CANON_R))


def i2(raw: IrisRaw) -> Result:
    try:
        px, py, _ = ic.pupil_blob(raw.grey)
    except ValueError as e:
        return failed(f"pupil: {e}")
    return _iris_result(raw, _canvas_map(px, py, ic.WORK_WIDTH / BASE))


def i3(raw: IrisRaw) -> Result:
    s, why = segment(raw)
    if s is None:
        return failed(why)
    return _iris_result(raw, RubberSheet(s["pupil"], s["iris"]), POLAR)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def i4(anchor_i3: Result, moving_i3: Result) -> Result:
    """I3, then the circular angular shift (+-16 samples) with the best correlation to the anchor."""
    if not (anchor_i3.ok and moving_i3.ok):
        return failed("I3 failed")
    shifts = range(-I4_SHIFT, I4_SHIFT + 1)
    r = [pearson(anchor_i3.out, np.roll(moving_i3.out, -s, axis=1)) for s in shifts]
    s = list(shifts)[int(np.nanargmax(r))]
    phi = Chain(Shift(s, POLAR[1]), moving_i3.phi)
    return Result(np.roll(moving_i3.out, -s, axis=1), phi, np.roll(moving_i3.valid, -s, axis=1), info={"shift": s})


def i5(anchor_frame: Result, moving_frame: Result, raw: IrisRaw) -> Result:
    if not (anchor_frame.ok and moving_frame.ok):
        return failed("frame failed")
    r = ecc(anchor_frame, moving_frame, BASE)
    if not r.ok:
        return r
    return _iris_result(raw, r.phi, info=r.info)


def h1_iris(frame: Result, ref_q: np.ndarray) -> Result:
    if not frame.ok:
        return frame
    out = frame.out.copy()
    out[frame.valid] = match_to_reference(out[frame.valid], ref_q)
    return Result(out, frame.phi, frame.valid)


# ---------- known motion (Q1a) ----------

def random_motion(rng: np.random.Generator, size: tuple, elastic: bool = False):
    """S: moved coords -> original coords (similarity about the centre, optional smooth field), for raw of `size`
    (h, w). Returns (S as a callable, parameters)."""
    h, w = size
    rot = np.radians(rng.uniform(-10, 10))
    sc = rng.uniform(0.9, 1.1)
    t = rng.uniform(-0.1, 0.1, 2) * w
    c = np.array([(w - 1) / 2, (h - 1) / 2])
    A = sc * np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    aff = Affine(np.vstack([np.hstack([A, (c + t - A @ c)[:, None]]), [0, 0, 1]]))
    if not elastic:
        return aff, {"rot": float(np.degrees(rot)), "scale": float(sc), "shift": t.tolist()}
    u = np.stack([ndimage.gaussian_filter(rng.standard_normal((h, w)), 16) for _ in (0, 1)], -1)
    u *= 4.0 / max(np.abs(u).max(), 1e-12)
    return Chain(Flow(u), aff), {"rot": float(np.degrees(rot)), "scale": float(sc), "shift": t.tolist(), "elastic": 4}


def move(raw: np.ndarray, S, shape=None) -> np.ndarray:
    """Moved image: moved(p) = raw(S(p))."""
    shape = shape or raw.shape[:2]
    if raw.ndim == 3:
        return np.stack([remap(raw[..., k], S, shape) for k in range(raw.shape[2])], -1)
    return remap(raw, S, shape)


def recovery_error(phi_anchor, phi_moved, S, out_pts: np.ndarray) -> np.ndarray:
    """Per point, raw-pixel distance between S(phi_moved(p)) and phi_anchor(p)."""
    return np.linalg.norm(S(phi_moved(out_pts)) - phi_anchor(out_pts), axis=1)


def grid16(valid: np.ndarray) -> np.ndarray:
    """16 x 16 grid of output points, kept where the anchor output is valid."""
    h, w = valid.shape
    ys, xs = np.linspace(0, h - 1, 18)[1:-1], np.linspace(0, w - 1, 18)[1:-1]
    pts = np.array([[x, y] for y in ys for x in xs])
    keep = valid[np.round(pts[:, 1]).astype(int), np.round(pts[:, 0]).astype(int)]
    return pts[keep]
