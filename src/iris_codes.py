"""Classical iris codes (Daugman-style) on UPOL and UTIRIS, calibrated on the identity ladder.

Pipeline, all classical and local (no learned model, no download):
  1. greyscale, resize to WORK_WIDTH, median filter against specular highlights;
  2. pupil: integro-differential operator (largest blurred radial derivative of the circular mean) over a grid of
     centres around the darkest region;
  3. limbus: the same operator on the lateral sectors only (eyelids cover the top and bottom), centres near the pupil;
  4. rubber-sheet normalisation between the two circles (RADIAL x ANGULAR samples);
  5. 1D log-Gabor filtering along the angle, phase quantised to 2 bits; bits masked where the eyelids usually sit
     and where intensity is extreme (lashes, reflections);
  6. fractional Hamming distance over bits valid in both codes, minimum over +-SHIFTS angular shifts (rotation).
Mirror: the fellow eye is compared after a horizontal flip of the whole image (the code is recomputed).
The audit of 2026-11-01 found about 35 degrees of freedom in these NIR codes (Daugman's systems have about 249) and
opposite flip effects in inner and outer radial bands, so the code measures identity on the ladder but is too weak to
test Daugman's fellow-eye result; the limbus centre search is also truncated in a share of images (reported).

Segmentation is checked numerically only (no image is displayed): radius ratio, centre offset and the share of
valid bits; images failing the checks are excluded and counted. Aggregate output: results/iris_codes.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from iris_calibration import calibrate, pairs_from_matrices, summarize, upol_table, utiris_table  # noqa: E402
from grape_calibration import TAUS  # noqa: E402,F401

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
WORK_WIDTH = 320
RADIAL, ANGULAR = 16, 128
SHIFTS = 6
N_THETA = 64
MIN_VALID_FRACTION = 0.25
EYELID_SECTORS_DEG = ((50, 130), (240, 300))  # upper and lower eyelid, angle counter-clockwise from +x with y up


def load_grey(path: str, flip: bool = False) -> np.ndarray:
    """Working image: the red channel for colour (visible-light) images, where melanin absorbs least and a dark iris
    separates from the pupil; the grey image otherwise (NIR)."""
    with Image.open(path) as im:
        im = im.getchannel("R") if im.mode in ("RGB", "RGBA") else im.convert("L")
        if flip:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        h = round(im.height * WORK_WIDTH / im.width)
        a = np.asarray(im.resize((WORK_WIDTH, h), Image.Resampling.BILINEAR), dtype=np.float64)
    return ndimage.median_filter(a, size=5)


def circle_means(img: np.ndarray, cx: np.ndarray, cy: np.ndarray, radii: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Mean intensity on circles: shape (n_centres, n_radii). Angles use y up: y = cy - r sin(theta)."""
    xs = cx[:, None, None] + radii[None, :, None] * np.cos(theta)[None, None, :]
    ys = cy[:, None, None] - radii[None, :, None] * np.sin(theta)[None, None, :]
    v = ndimage.map_coordinates(img, [ys.ravel(), xs.ravel()], order=1, mode="nearest")
    return v.reshape(xs.shape).mean(2)


def integro_differential(img, cx, cy, radii, theta, sigma: float = 1.5):
    """Best (centre index, radius, score) by the blurred radial derivative of the circular mean (dark inside)."""
    m = circle_means(img, cx, cy, radii, theta)
    d = ndimage.gaussian_filter1d(np.gradient(m, radii, axis=1), sigma, axis=1)
    k = np.unravel_index(int(np.argmax(d)), d.shape)
    return k[0], float(radii[k[1]]), float(d[k])


def pupil_blob(img: np.ndarray) -> tuple[float, float, float]:
    """Pupil candidate: the most disk-like large dark component (centre x, centre y, equivalent radius).

    The strongest circular edge in an eye image is often the limbus, not the pupil, so the pupil is located as a dark
    blob first and the integro-differential operator only refines it."""
    h, w = img.shape
    content = img[img > 2]  # some sets (UPOL) frame the eye with exact black, which is not tissue
    if content.size < 0.1 * img.size:
        raise ValueError("image is almost empty")
    lo, med = np.percentile(content, 0.5), np.median(content)
    for q in (0.15, 0.25, 0.35, 0.5):  # darkest threshold first: the pupil is darker than any iris crypt
        lab, n = ndimage.label((img < lo + q * (med - lo)) & (img > 2))
        best = None
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w:
                continue  # the pupil never touches the frame
            comp = lab[sl] == k
            area = comp.sum()
            if area < max(20, 5e-4 * h * w) or area > 0.25 * h * w:
                continue
            fill = area / (np.pi * (max(comp.shape) / 2) ** 2)
            if fill < 0.6:
                continue
            if best is None or area > best[0]:
                cy, cx = ndimage.center_of_mass(comp)
                best = (area, sl[1].start + cx, sl[0].start + cy, float(np.sqrt(area / np.pi)))
        if best is not None:
            return best[1], best[2], best[3]
    raise ValueError("no disk-like dark region for the pupil")


def segment(img: np.ndarray) -> dict:
    h, w = img.shape
    full = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    bx, by, br = pupil_blob(img)
    fx, fy = np.meshgrid(np.arange(bx - 3, bx + 4, 1.0), np.arange(by - 3, by + 4, 1.0))
    j, rp, sp = integro_differential(img, fx.ravel(), fy.ravel(), np.arange(max(3, 0.7 * br), 1.3 * br + 1, 0.5), full)
    px, py = fx.ravel()[j], fy.ravel()[j]
    lateral = np.concatenate([np.linspace(-np.pi / 4, np.pi / 4, N_THETA // 2), np.linspace(3 * np.pi / 4, 5 * np.pi / 4, N_THETA // 2)])
    lx, ly = np.meshgrid(np.arange(px - 6, px + 7, 2.0), np.arange(py - 6, py + 7, 2.0))
    ri_range = np.arange(1.5 * rp, min(5.0 * rp, 0.48 * w), 1.0)
    if len(ri_range) < 5:
        raise ValueError("no room for the limbus search")
    k, ri, si = integro_differential(img, lx.ravel(), ly.ravel(), ri_range, lateral)
    return {"pupil": (px, py, rp), "iris": (lx.ravel()[k], ly.ravel()[k], ri), "ratio": rp / ri,
            "offset": float(np.hypot(lx.ravel()[k] - px, ly.ravel()[k] - py) / ri), "pupil_score": sp, "iris_score": si}


def normalise(img: np.ndarray, seg: dict) -> np.ndarray:
    """Rubber sheet: RADIAL x ANGULAR samples, linear between the pupil and limbus boundary points at each angle."""
    (px, py, rp), (ix, iy, ri) = seg["pupil"], seg["iris"]
    th = np.linspace(0, 2 * np.pi, ANGULAR, endpoint=False)
    bx_p, by_p = px + rp * np.cos(th), py - rp * np.sin(th)
    bx_i, by_i = ix + ri * np.cos(th), iy - ri * np.sin(th)
    t = np.linspace(0.08, 0.92, RADIAL)[:, None]  # stay off both boundaries
    xs, ys = bx_p + t * (bx_i - bx_p), by_p + t * (by_i - by_p)
    return ndimage.map_coordinates(img, [ys, xs], order=1, mode="nearest")


def log_gabor_code(polar: np.ndarray, wavelength: float = 16.0, sigma_on_f: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """2-bit phase code per sample from a 1D log-Gabor filter along the angle; returns (bits, valid mask)."""
    n = polar.shape[1]
    f = np.fft.fftfreq(n)
    f0 = 1.0 / wavelength
    G = np.zeros(n)
    pos = f > 0
    G[pos] = np.exp(-(np.log(f[pos] / f0) ** 2) / (2 * np.log(sigma_on_f) ** 2))
    resp = np.fft.ifft(np.fft.fft(polar - polar.mean(1, keepdims=True), axis=1) * G, axis=1)
    bits = np.stack([resp.real > 0, resp.imag > 0], axis=-1)
    lo, hi = np.percentile(polar, [3, 97])
    valid = (polar > lo) & (polar < hi)
    th = np.degrees(np.linspace(0, 2 * np.pi, n, endpoint=False))
    for a, b in EYELID_SECTORS_DEG:
        valid[:, (th >= a) & (th <= b)] = False
    valid &= np.abs(resp) > 1e-9
    return bits, np.repeat(valid[..., None], 2, axis=-1)


def hamming_matrix(codes: np.ndarray, masks: np.ndarray, codes_b: np.ndarray | None = None, masks_b: np.ndarray | None = None,
                   shifts: int = SHIFTS) -> np.ndarray:
    """Fractional Hamming distance between every code in A and every code in B, minimum over angular shifts.

    codes/masks: (n, RADIAL, ANGULAR, 2) booleans. Uses disagreements = A.M' + M.A' - 2 A.A' on masked bits."""
    codes_b = codes if codes_b is None else codes_b
    masks_b = masks if masks_b is None else masks_b
    A = (codes & masks).reshape(len(codes), -1).astype(np.float32)
    M = masks.reshape(len(masks), -1).astype(np.float32)
    best = np.full((len(codes), len(codes_b)), np.inf, dtype=np.float32)
    for s in range(-shifts, shifts + 1):
        cb, mb = np.roll(codes_b, s, axis=2), np.roll(masks_b, s, axis=2)
        B = (cb & mb).reshape(len(cb), -1).astype(np.float32)
        N = mb.reshape(len(mb), -1).astype(np.float32)
        valid = M @ N.T
        dis = A @ N.T + M @ B.T - 2 * (A @ B.T)
        frac = valid / (RADIAL * ANGULAR * 2)
        hd = np.where(frac >= MIN_VALID_FRACTION ** 2, dis / np.maximum(valid, 1), np.inf)
        best = np.minimum(best, hd)
    return np.where(np.isfinite(best), best, np.nan).astype(np.float64)


def encode(paths, flip: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    codes, masks, qc = [], [], []
    for p in paths:
        img = load_grey(p, flip)
        try:
            s = segment(img)
            b, v = log_gabor_code(normalise(img, s))
            ok = 0.15 <= s["ratio"] <= 0.75 and s["offset"] <= 0.25 and v.mean() >= MIN_VALID_FRACTION
            qc.append({"ratio": s["ratio"], "offset": s["offset"], "valid_bits": float(v.mean()), "ok": ok, "error": ""})
        except ValueError as e:
            b = np.zeros((RADIAL, ANGULAR, 2), bool)
            v = np.zeros_like(b)
            qc.append({"ratio": np.nan, "offset": np.nan, "valid_bits": 0.0, "ok": False, "error": type(e).__name__})
        if not qc[-1]["ok"]:
            v = np.zeros_like(v)  # an image that fails segmentation QC gets no valid bits, so its distances are NaN
        codes.append(b)
        masks.append(v)
    return np.stack(codes), np.stack(masks), pd.DataFrame(qc)


def dprime(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(abs(y.mean() - x.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))


def run(t: pd.DataFrame) -> dict:
    t = t.reset_index(drop=True)
    C, Mk, qc = encode(t.path)
    Cf, Mf, qcf = encode(t.path, flip=True)
    D = hamming_matrix(C, Mk)
    Dm = hamming_matrix(C, Mk, Cf, Mf)
    p = calibrate(pairs_from_matrices(t, D, Dm))
    p_plain = calibrate(pairs_from_matrices(t, D, None))
    hd = {r: g.distance for r, g in p_plain.groupby("rung")}  # raw Hamming distances without flipping, one reference
    out = {"n_images": int(len(t)), "segmentation_qc_pass": float(qc.ok.mean()), "flipped_qc_pass": float(qcf.ok.mean()),
           "qc_median_among_segmented": {"pupil_iris_ratio": float(qc.ratio.median()), "centre_offset": float(qc.offset.median()),
                                         "valid_bits": float(qc.loc[qc.valid_bits > 0, "valid_bits"].median())},
           "median_hamming": {r: float(v.median()) for r, v in hd.items()},
           "decidability_vs_other_person": {r: dprime(v, hd["other_person"]) for r, v in hd.items() if r != "other_person" and len(v) > 2},
           "calibrated_fellow_eye_flipped": summarize(p), "calibrated_fellow_eye_plain": summarize(p_plain)}
    out["note"] = ("median_hamming and decidability use unflipped codes; calibrated scores use a reference matched on "
                   "sensor pair and eye relation. Iris-code degrees of freedom are far below Daugman's, so this code "
                   "cannot test his fellow-eye result.")
    return out


def main():
    up = upol_table()
    ut, conflicts = utiris_table()
    out = {"method": "integro-differential segmentation, rubber sheet 16x128, 1D log-Gabor, 2-bit phase, "
                     f"Hamming distance min over +-{SHIFTS} shifts; red channel for colour images",
           "reference": "unrelated-person pairs in the same sensor and eye condition",
           "primary": "UTIRIS_NIR (segmentation is reliable in NIR); visible-light subsets are secondary because classical "
                      "segmentation passes QC for a minority of visible-light images",
           "UTIRIS_NIR": {"excluded_eye_label_conflicts": conflicts, **run(ut[ut.sensor == "NIR"])},
           "UTIRIS_VIS_secondary": run(ut[ut.sensor == "VIS"]),
           "UTIRIS_NIR_vs_VIS_secondary": run(ut),
           "UPOL_secondary": run(up)}
    with open(os.path.join(PROJECT, "results", "iris_codes.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for ds in ("UTIRIS_NIR", "UTIRIS_VIS_secondary", "UTIRIS_NIR_vs_VIS_secondary", "UPOL_secondary"):
        o = out[ds]
        print(ds, "QC pass", round(o["segmentation_qc_pass"], 3), "median HD", {k: round(v, 3) for k, v in o["median_hamming"].items()})
        print("  c (flipped fellow eye)", {k: round(v["mean_c_auc"], 3) for k, v in o["calibrated_fellow_eye_flipped"].items()})
        print("  c (plain fellow eye)", {k: round(v["mean_c_auc"], 3) for k, v in o["calibrated_fellow_eye_plain"].items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
