"""Registration study runner (docs/PROTOCOL_registration.md, Amendments 1 to 4).

  python src/reg_run.py prep   --dataset GRAPE|UTIRIS|UPOL      raw arrays, vessel maps, iris structures (private)
  python src/reg_run.py run    --dataset GRAPE --methods G0,G1   one experiment per method (private records)
  python src/reg_run.py report                                  aggregates and the comparison table (public)

Private outputs (image names, person IDs, per-pair values) stay in RUN_DIR; results/registration/ gets aggregates.
Smoke runs (--limit) never write results.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import grape_calibration as gc  # noqa: E402
import iris_codes as ic  # noqa: E402
import ladder2d as L  # noqa: E402
import ladder2d_register as R  # noqa: E402
import ladder2d_select as sel  # noqa: E402
import reg_eval as E  # noqa: E402
import reg_methods as M  # noqa: E402

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PROTOCOL = os.path.join(PROJECT, "docs", "PROTOCOL_registration.md")
RUN_DIR = os.path.join(sel.DATA_ROOT, "projects", "identity-ladder-metric", "run-2026-11-02-registration")
RES_DIR = os.path.join(PROJECT, "results", "registration")
SEED = 20261101
WORKERS = 4
N_HREF = 50
Q1D_STRANGERS = 5
METHODS = {"GRAPE": ["G0", "G1", "G2", "G3", "G4", "H1", "H2"],
           "UTIRIS": ["I0", "I1", "I2", "I3", "I4", "I5", "H1"], "UPOL": ["I0", "I1", "I2", "I3", "I4", "I5", "H1"]}
PAIRWISE = {"G2", "G3", "G4", "I4", "I5"}
IRIS_FRAME = {"UTIRIS": "I1", "UPOL": "I2"}  # frame under I5 and iris H1


def frame_of(ds: str, m: str) -> str:
    return IRIS_FRAME[ds] if ds != "GRAPE" and m in ("I5", "H1") else ""


def text_sha256(path: str) -> str:
    """Line-ending independent hash (the checkout may convert LF to CRLF)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


# ---------- selections and private stores ----------

def selection(ds: str) -> list:
    key = "selection_grape" if ds == "GRAPE" else "selection_iris"
    pairs = json.load(open(L.EXPECTED[key][0], encoding="utf-8"))["pairs"]
    return [p for p in pairs if p.get("dataset", "GRAPE") == ds]


def image_names(pairs: list) -> list:
    return sorted({n for p in pairs for n in [p["anchor"], p["candidate"]] + p["reference"] + p["mean_members"]})


def image_path(ds: str, name: str) -> str:
    return os.path.join({"GRAPE": L.GRAPE_IMG, "UTIRIS": sel.UTIRIS, "UPOL": sel.UPOL}[ds], name)


CACHE_DIR = os.path.join(os.path.dirname(sel.DATA_ROOT), "cache", "identity-ladder-registration")  # per machine, not synced


def store(ds: str) -> str:
    return os.path.join(CACHE_DIR, f"store_{ds}")


def prep(ds: str) -> None:
    hashes = L.guard()
    pairs = selection(ds)
    names = image_names(pairs)
    os.makedirs(store(ds), exist_ok=True)
    img_hash = {n: L.sha256(image_path(ds, n)) for n in names}
    if ds == "GRAPE":
        raw = np.stack([(L.load_base(image_path(ds, n), M.G_SIZE, True) * 255).astype(np.uint8) for n in names])
        np.save(os.path.join(store(ds), "raw.npy"), raw)
        ves = np.stack([np.packbits(L.vessel_masks(r.astype(np.float32) / 255)[0]) for r in raw])
        np.save(os.path.join(store(ds), "vessel.npy"), ves)
        u = sel.grape_universe(pd.read_csv(sel.GRAPE_META)).set_index("image")
        extra = {"camera": {n: str(u.loc[n, "device"]) for n in names}}
    else:
        arrs, struct = {}, {}
        for i, n in enumerate(names):
            for flip in (False, True):
                r = M.iris_raw(image_path(ds, n), flip)
                if not flip:
                    arrs[f"g{i}"], arrs[f"l{i}"] = r.grey.astype(np.float32), r.luma.astype(np.float32)
                    arrs[f"o{i}"] = np.array(r.orig)
                struct[f"{n}|{int(flip)}"] = iris_structures(r)
        np.savez(os.path.join(store(ds), "iris.npz"), **arrs)
        extra = {"structures": struct}
    with open(os.path.join(store(ds), "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"names": names, "image_sha256": img_hash, "inputs": hashes, **extra}, fh)
    print(f"{ds}: {len(names)} images cached")


def iris_structures(r: "M.IrisRaw") -> dict:
    out = {}
    try:
        px, py, _ = ic.pupil_blob(r.grey)
        out["pupil"] = [float(px), float(py)]
    except ValueError:
        pass
    s, _ = M.segment(r)
    if s is not None:
        out["limbus"] = [float(s["iris"][0]), float(s["iris"][1])]
        out["mid_r"] = float((s["pupil"][2] + s["iris"][2]) / 2)
    return out


# ---------- worker state ----------

W = {}


def init_worker(ds: str, methods: list, aux: dict) -> None:
    M.cv2.setNumThreads(1)
    idx = json.load(open(os.path.join(store(ds), "index.json"), encoding="utf-8"))
    W.update(ds=ds, methods=methods, aux=aux, idx={n: i for i, n in enumerate(idx["names"])}, index=idx, cache={})
    if ds == "GRAPE":
        W["raw"] = np.load(os.path.join(store(ds), "raw.npy"), mmap_mode="r")
        W["ves"] = np.load(os.path.join(store(ds), "vessel.npy"), mmap_mode="r")
    else:
        W["npz"] = np.load(os.path.join(store(ds), "iris.npz"))


def raw(name: str, flip: bool):
    i = W["idx"][name]
    if W["ds"] == "GRAPE":
        a = W["raw"][i].astype(np.float32) / 255
        return a[:, ::-1].copy() if flip else a
    g, lu, o = (W["npz"][f"{k}{i}"] for k in "glo")
    if flip:
        g, lu = g[:, ::-1].copy(), lu[:, ::-1].copy()
    return M.IrisRaw(g.astype(np.float64), lu.astype(np.float64), tuple(int(x) for x in o))


def vessel(name: str, flip: bool) -> np.ndarray:
    v = np.unpackbits(W["ves"][W["idx"][name]])[: M.G_SIZE * M.G_SIZE].reshape(M.G_SIZE, M.G_SIZE).astype(bool)
    return v[:, ::-1] if flip else v


def frame(m: str, r) -> "M.Result":
    """Per-image method (or the frame under a pairwise method) applied to a raw image."""
    ds, aux = W["ds"], W["aux"]
    if ds == "GRAPE":
        if m in ("G0", "G2"):
            return M.g0(r)
        g = M.g1(r)
        if m == "H1":
            return M.h1_fundus(g, aux["href"])
        if m == "H2":
            return g  # standardised later with the camera of the image (needs its name)
        return g
    f = {"I0": M.i0, "I1": M.i1, "I2": M.i2, "I3": M.i3, "I4": M.i3}.get(m)
    if f is not None:
        return f(r)
    base = M.i1(r) if IRIS_FRAME[ds] == "I1" else M.i2(r)
    return M.h1_iris(base, aux["href"]) if m == "H1" else base


def slot(m: str, A: "M.Result", A_raw, name: str, flip: bool, r=None) -> "M.Result":
    """The candidate-slot image `name` through method m, given the anchor's frame result A."""
    r = raw(name, flip) if r is None else r
    if m not in PAIRWISE:
        res = frame(m, r)
        if m == "H2" and res.ok:
            res = M.h2_fundus(res, W["aux"]["cam"][W["index"]["camera"][name]])
        return res
    if m == "G2":
        return M.g2(A, r)
    if m in ("G3", "G4"):
        g1r = M.g1(r)
        g3r = M.g3(A, g1r, r) if g1r.ok else M.failed("G1 failed")
        return g3r if m == "G3" else M.g4(A, g3r, r)
    if m == "I4":
        return M.i4(A, M.i3(r))
    return M.i5(A, frame("I5", r), r)


def anchor_frame(m: str, name: str, flip: bool = False):
    r = raw(name, flip)
    A = frame(m, r)
    if m == "H2" and A.ok:
        A = M.h2_fundus(A, W["aux"]["cam"][W["index"]["camera"][name]])
    return A, r


# ---------- features of one output ----------

def grey224(res: "M.Result") -> tuple:
    """(grey analysis image, valid fraction image) on the 224 grid (iris I3/I4: the 64 x 256 unwrapped image)."""
    if W["ds"] == "GRAPE":
        g = L.box_resize(L.grey(res.out), M.BASE)
        v = L.box_resize(res.valid.astype(np.float32), M.BASE)
        return g, v
    return np.asarray(res.out, np.float64), res.valid.astype(np.float64)


def patch(m: str, a: np.ndarray) -> np.ndarray:
    if W["ds"] == "GRAPE" or m == "I0":
        return a[80:144, 80:144]
    if m in ("I3", "I4"):
        return a
    return a[R.IRIS_BOX]


def rep_P(m: str, g: np.ndarray) -> np.ndarray:
    return L.zscore(L.box_resize(g, L.P_SIZE).ravel())


def rep_Nraw(g: np.ndarray) -> np.ndarray:
    return np.array([g.mean(), g.std()])


def landmarks(res: "M.Result"):
    if W["ds"] != "GRAPE" or res.out is None or W["m"] in ("H1", "H2"):
        return None
    return R.fundus_landmarks(res.out)


def features(m: str, res: "M.Result", name: str, flip: bool) -> dict:
    g, v = grey224(res)
    f = {"g": g, "v": v, "P": rep_P(m, g), "N": rep_Nraw(g), "patch": patch(m, g), "cov": float(patch(m, v).mean())}
    if W["ds"] == "GRAPE":
        f["ves"] = M.remap(vessel(name, flip).astype(np.float32), res.phi, (M.G_SIZE, M.G_SIZE), nearest=True) > 0.5
        f["lm"] = landmarks(res) if m in ("G0", "G1", "G2", "G3", "G4") else None
    f["phi"] = res.phi
    f["info"] = res.info
    return f


def cached_features(m: str, name: str, flip: bool):
    """Per-image methods: the slot output does not depend on the anchor, so its features are cached."""
    key = (m, name, flip)
    if key not in W["cache"]:
        res = slot(m, None, None, name, flip)
        W["cache"][key] = (res.fail, features(m, res, name, flip) if res.ok else None)
    return W["cache"][key]


# ---------- Q1b ----------

def q1b(m: str, fa: dict, fs: dict, a_name: str, a_flip: bool, s_name: str, s_flip: bool) -> dict:
    out = {}
    if W["ds"] == "GRAPE":
        lmA = fa.get("lm_anchor")
        if lmA is not None:
            ring = fa["ring"]
            a, b = fa["ves"] & ring, fs["ves"] & ring
            s = a.sum() + b.sum()
            out["dice"] = float(2 * (a & b).sum() / s) if s else np.nan
            if fs.get("lm") is not None:
                k = M.BASE / M.G_SIZE
                out["disc"] = float(np.linalg.norm(fs["lm"][0] - lmA[0]) * k)
                out["fovea"] = float(np.linalg.norm(fs["lm"][1] - lmA[1]) * k)
        return out
    if m in ("I3", "I4"):
        return out
    st = W["index"]["structures"]
    sa, ss = st[f"{a_name}|{int(a_flip)}"], st[f"{s_name}|{int(s_flip)}"]
    k = M.BASE / ic.WORK_WIDTH
    for key in ("pupil", "limbus"):
        if key in sa and key in ss:
            p_out = fs["phi"].inverse()(np.array([ss[key]]))
            p_anchor_raw = fa["phi"](p_out)[0]
            out[key] = float(np.linalg.norm(p_anchor_raw - np.array(sa[key])) * k)
    return out


# ---------- one pair ----------

def standardise_N(v: np.ndarray) -> np.ndarray:
    mu, sd = W["aux"]["n_stats"]
    return (v - np.array(mu)) / np.array(sd)


def dist(fa, fs, rep):
    if rep == "P":
        return L.euclid(fa["P"], fs["P"])
    return L.euclid(standardise_N(fa["N"]), standardise_N(fs["N"]))


def slot_features(m, A, A_raw, name, flip, r=None):
    if m not in PAIRWISE and r is None:
        return cached_features(m, name, flip)
    res = slot(m, A, A_raw, name, flip, r)
    return res.fail, (features(m, res, name, flip) if res.ok else None)


def anchor_features(m, A, name, flip):
    fa = features(m, A, name, flip)
    if W["ds"] == "GRAPE":
        fa["lm_anchor"] = R.fundus_landmarks(A.out) if m not in ("H1", "H2") else R.fundus_landmarks(M.g1(raw(name, flip)).out)
        fa["valid512"] = A.valid
        if fa["lm_anchor"] is not None:
            r_fov = np.sqrt(A.valid.sum() / np.pi)
            yy, xx = np.mgrid[:M.G_SIZE, :M.G_SIZE]
            d = np.hypot(xx - fa["lm_anchor"][0][0], yy - fa["lm_anchor"][0][1])
            fa["ring"] = (d >= 0.3 * r_fov) & (d <= 0.6 * r_fov) & A.valid
    return fa


def c_of(fa, fs, ref, rep="P"):
    return float(gc.calibrated_score(np.array([dist(fa, fs, rep)]), np.array([dist(fa, s, rep) for s in ref]))[0])


def mean_features(m, fs_list):
    g = np.mean([f["g"] for f in fs_list], axis=0)
    return {"P": rep_P(m, g), "N": rep_Nraw(g)}


def inverse_consistency(m, A, A_raw, fa, a_name, a_flip, s_name, s_flip, fs) -> float:
    """Q1d: register the slot onto the anchor and back; raw-pixel distance of the round trip (224 grid)."""
    if m == "I4":
        B, _ = anchor_frame(m, s_name, s_flip)
        if not B.ok:
            return np.nan
        back = slot(m, B, None, a_name, a_flip)
        if not back.ok:
            return np.nan
        mid_r = W["index"]["structures"][f"{a_name}|{int(a_flip)}"].get("mid_r", np.nan)
        return abs(fs["info"]["shift"] + back.info["shift"]) * 2 * np.pi * mid_r / M.POLAR[1] * M.BASE / ic.WORK_WIDTH
    B, B_raw = anchor_frame(m, s_name, s_flip)
    if not B.ok or not isinstance(B.phi, M.Affine):
        return np.nan
    back = slot(m, B, B_raw, a_name, a_flip)
    if not back.ok:
        return np.nan
    pts = M.grid16(A.valid)
    there = fs["phi"](pts)
    round_trip = back.phi(B.phi.inverse()(there))
    k = M.BASE / (M.G_SIZE if W["ds"] == "GRAPE" else ic.WORK_WIDTH)
    return float(np.median(np.linalg.norm(round_trip - A.phi(pts), axis=1)) * k)


def process(args) -> list:
    i, p = args
    rng = np.random.default_rng(SEED + i)
    null = p["reference"][int(rng.integers(len(p["reference"])))]
    fel = p["rung"].startswith("fellow")
    recs = []
    for m in W["methods"]:
        W["m"] = m
        W["cache"] = {k: v for k, v in W["cache"].items() if isinstance(k[0], str) and k[0] == m}  # per-image only
        rec = {"pair": i, "person": p["person"], "rung": p["rung"], "method": m, "dropped": "",
               "fail": {"anchor": False, "candidate": False, "strangers": [0, len(p["reference"])],
                        "members": [0, len(p["mean_members"])]},
               "c": {}, "q1b": {}, "q1c": {}, "coverage": {}, "coverage_lt95": {}, "q1d": {}, "q1e": {},
               "slots": {}}
        A, A_raw = anchor_frame(m, p["anchor"])
        if not A.ok:
            rec["fail"]["anchor"] = True
            rec["dropped"] = "anchor"
            recs.append(rec)
            continue
        fa = anchor_features(m, A, p["anchor"], False)
        fc_fail, fc = slot_features(m, A, A_raw, p["candidate"], fel)
        rec["fail"]["candidate"] = bool(fc_fail)
        fs = {}
        for s in p["reference"]:
            why, f = slot_features(m, A, A_raw, s, fel)
            if why:
                rec["fail"]["strangers"][0] += 1
            else:
                fs[s] = f
        fm, fm_names = [], []
        for s in p["mean_members"]:
            why, f = slot_features(m, A, A_raw, s, fel)
            if why:
                rec["fail"]["members"][0] += 1
            else:
                fm.append(f)
                fm_names.append(s)
        ref = [f for s, f in fs.items() if s != null]
        rec["slots"] = {"strangers_ok": [s for s in p["reference"] if s in fs]}
        if fc is None:
            rec["dropped"] = "candidate"
        elif len(ref) < 30:
            rec["dropped"] = "few strangers"
        elif len(fm) < 8:
            rec["dropped"] = "few members"
        if rec["dropped"]:
            recs.append(rec)
            continue
        fmean = mean_features(m, fm)
        for rep in ("P", "N"):
            rec["c"][f"own_{rep}"] = c_of(fa, fc, ref, rep)
            rec["c"][f"mean_{rep}"] = c_of(fa, fmean, ref, rep)
        rec["c"]["null_P"] = c_of(fa, fs[null], ref) if null in fs else np.nan
        # Q3d: anchor and candidate exchanged, same strangers re-registered to the new anchor
        A2, A2_raw = anchor_frame(m, p["candidate"])
        if A2.ok:
            fa2 = anchor_features(m, A2, p["candidate"], False)
            _, fc2 = slot_features(m, A2, A2_raw, p["anchor"], fel)
            ref2 = [f for f in (slot_features(m, A2, A2_raw, s, False)[1] for s in fs if s != null) if f is not None]
            rec["c"]["swap_P"] = c_of(fa2, fc2, ref2) if fc2 is not None and len(ref2) >= 30 else np.nan
        else:
            rec["c"]["swap_P"] = np.nan
        # Q3e: reflected candidate, reflection undone by exact index reversal before the method
        if not fel:
            r = raw(p["candidate"], True)
            r = r[:, ::-1].copy() if W["ds"] == "GRAPE" else M.IrisRaw(r.grey[:, ::-1].copy(), r.luma[:, ::-1].copy(), r.orig)
            res = slot(m, A, A_raw, p["candidate"], False, r) if m in PAIRWISE else frame(m, r)
            if m == "H2" and res.ok:
                res = M.h2_fundus(res, W["aux"]["cam"][W["index"]["camera"][p["candidate"]]])
            rec["c"]["reflect_P"] = c_of(fa, features(m, res, p["candidate"], False), ref) if res.ok else np.nan
        # Q1b, Q1c, coverage per role
        roles = {"candidate": [(p["candidate"], fc)], "strangers": [(s, fs[s]) for s in fs if s != null],
                 "members": list(zip(fm_names, fm)), "null": [(null, fs.get(null))]}
        for role, items in roles.items():
            items = [(n, f) for n, f in items if f is not None]
            if not items:
                continue
            b = [q1b(m, fa, f, p["anchor"], False, n, fel) for n, f in items]
            rec["q1b"][role] = {k: float(np.nanmedian([x.get(k, np.nan) for x in b]))
                                for k in ("dice", "disc", "fovea", "pupil", "limbus")
                                if any(k in x for x in b)}
            rec["q1c"][role] = float(np.nanmedian([M.pearson(fa["patch"], f["patch"]) for _, f in items]))
            covs = [f["cov"] for _, f in items]
            rec["coverage"][role] = float(np.median(covs))
            rec["coverage_lt95"][role] = float(np.mean(np.array(covs) < 0.95))
        if m in PAIRWISE:
            first = [s for s in p["reference"] if s in fs and s != null][:Q1D_STRANGERS]
            rec["q1d"]["candidate"] = inverse_consistency(m, A, A_raw, fa, p["anchor"], False, p["candidate"], fel, fc)
            rec["q1d"]["null"] = inverse_consistency(m, A, A_raw, fa, p["anchor"], False, null, fel, fs[null]) \
                if null in fs else np.nan
            rec["q1d"]["strangers"] = float(np.nanmedian(
                [inverse_consistency(m, A, A_raw, fa, p["anchor"], False, s, fel, fs[s]) for s in first])) \
                if first else np.nan
        if m == "G4":
            rec["q1e"]["candidate"] = fc["info"].get("jac_p99_abslog", np.nan)
            rec["q1e"]["strangers"] = float(np.nanmedian([f["info"].get("jac_p99_abslog", np.nan)
                                                          for s, f in fs.items() if s != null]))
        rec["slots"]["members_ok"] = fm_names
        recs.append(rec)
    return recs


# ---------- Q1a per image ----------

def q1a_image(args) -> dict:
    j, name = args
    out = {}
    for m in W["methods"]:
        W["m"] = m
        rng = np.random.default_rng(SEED + 100000 + j)
        r = raw(name, False)
        shape = r.shape[:2] if W["ds"] == "GRAPE" else r.luma.shape
        S, _ = M.random_motion(rng, shape, elastic=(m == "G4"))
        moved = M.move(r, S) if W["ds"] == "GRAPE" else M.IrisRaw(M.move(r.grey, S), M.move(r.luma, S), r.orig)
        A = frame(m, r)
        if not A.ok:
            out[m] = np.nan
            continue
        if m in PAIRWISE:
            B = slot(m, A, r, name, False, moved)
        else:
            B = frame(m, moved)
        if not B.ok:
            out[m] = np.nan
            continue
        k = M.BASE / (M.G_SIZE if W["ds"] == "GRAPE" else ic.WORK_WIDTH)
        out[m] = float(np.median(M.recovery_error(A.phi, B.phi, S, M.grid16(A.valid))) * k)
    return {"image": name, "q1a": out}


# ---------- run ----------

def aux_for(ds: str, methods: list, pairs: list) -> dict:
    """Harmonisation references and N statistics, computed in the parent once (Amendment 1)."""
    init_worker(ds, methods, {})
    aux = {}
    names = W["index"]["names"]
    rng = np.random.default_rng(SEED)
    pool = [names[k] for k in rng.choice(len(names), size=min(N_HREF, len(names)), replace=False)]
    if ds == "GRAPE":
        g1s = [M.g1(raw(n, False)) for n in pool]
        g1s = [g for g in g1s if g.ok]
        aux["href"] = [M.quantile_reference(np.concatenate([g.out[..., c][g.valid] for g in g1s])) for c in range(3)]
        cams = {}
        for n, g in zip([n for n in pool], [M.g1(raw(n, False)) for n in pool]):
            if g.ok:
                cams.setdefault(W["index"]["camera"][n], []).append(g.out[g.valid])
        aux["cam"] = {c: {"mean": np.concatenate(v).mean(0).tolist(), "sd": np.concatenate(v).std(0).tolist()}
                      for c, v in cams.items()}
        missing = set(W["index"]["camera"].values()) - set(aux["cam"])
        if missing:
            raise SystemExit(f"H2 camera without reference images: {sorted(missing)}")
        aux["href_n"] = len(g1s)
    else:
        base = M.i1 if IRIS_FRAME[ds] == "I1" else M.i2
        fr = [base(raw(n, False)) for n in pool]
        fr = [f for f in fr if f.ok]
        aux["href"] = M.quantile_reference(np.concatenate([f.out[f.valid] for f in fr]))
        aux["href_n"] = len(fr)
    W["aux"] = aux
    return aux


def n_stats(ds: str, m: str, pairs: list, aux: dict) -> list:
    init_worker(ds, [m], aux)
    W["m"] = m
    vals = []
    for n in sorted({p["anchor"] for p in pairs}):
        A, _ = anchor_frame(m, n)
        if A.ok:
            vals.append(rep_Nraw(grey224(A)[0]))
    v = np.array(vals)
    return [v.mean(0).tolist(), v.std(0).tolist()]


def run(ds: str, methods: list, limit: int | None) -> None:
    t0 = time.time()
    hashes = L.guard()
    pairs = selection(ds)
    if limit:
        pairs = pairs[:limit]
    aux = aux_for(ds, methods, pairs)
    for m in methods:
        tm = time.time()
        aux_m = {**aux, "n_stats": n_stats(ds, m, pairs, aux)}
        with Pool(WORKERS, initializer=init_worker, initargs=(ds, [m], aux_m)) as pool:
            names = image_names(pairs)
            q1a = {r["image"]: r["q1a"][m] for r in pool.imap(q1a_image, list(enumerate(names)), chunksize=8)}
            recs = [r for rs in pool.imap(process, list(enumerate(pairs)), chunksize=2) for r in rs]
        for r in recs:
            sl = r.pop("slots")
            if r["fail"]["anchor"] or r["dropped"]:
                r["q1a"] = {}
                continue
            p = pairs[r["pair"]]
            r["q1a"] = {"candidate": q1a.get(p["candidate"], np.nan),
                        "strangers": float(np.nanmedian([q1a.get(s, np.nan) for s in sl["strangers_ok"]] or [np.nan])),
                        "members": float(np.nanmedian([q1a.get(s, np.nan) for s in sl.get("members_ok", [])] or [np.nan])),
                        "null": np.nan}
        if limit:
            print(f"smoke {ds} {m}: {len(recs)} records, {time.time() - tm:.0f} s; nothing written")
            continue
        out = os.path.join(RUN_DIR, f"{ds}_{m}")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "records.json"), "w", encoding="utf-8") as fh:
            json.dump({"records": recs, "q1a": q1a, "aux_href_n": aux["href_n"]}, fh, default=float)
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as fh:
            json.dump({"protocol_sha256": text_sha256(PROTOCOL), "inputs": hashes, "seconds": time.time() - tm,
                       "code_sha256": {f: text_sha256(os.path.join(PROJECT, "src", f))
                                       for f in ("reg_run.py", "reg_methods.py", "reg_eval.py",
                                                 "ladder2d_register.py", "iris_codes.py", "ladder2d.py")}}, fh)
        print(f"{ds} {m}: {len(recs)} records in {time.time() - tm:.0f} s", flush=True)
    print(f"total {time.time() - t0:.0f} s")


# ---------- report ----------

def summarise(ds: str, m: str, recs: list, none_recs: list | None) -> dict:
    fr = frame_of(ds, m)
    fail = E.failure_rates(recs)
    fair = E.fairness(recs, m, fr)
    acc = E.accuracy(recs, m, fr)
    elig = E.eligibility(fail, fair, m, fr)
    key, rank, higher = E.ranking_value(acc, ds, m, fr)
    out = {"dataset": ds, "method": m, "frame": fr, "n_pairs": len(recs),
           "n_dropped": {k: sum(r["dropped"] == k for r in recs) for k in ("anchor", "candidate", "few strangers",
                                                                         "few members")},
           "failure": fail, "accuracy": acc, "fairness": fair, "eligibility": elig,
           "rank": {"key": key, "value": rank, "higher_is_better": higher},
           "downstream": E.downstream(recs, ds)}
    if none_recs is not None and m not in ("G0", "I0"):
        out["downstream"].update(E.versus_none(recs, none_recs, ds))
    return out


def fmt(s, k=2):
    if s is None or s.get("mean") is None:
        return "n/a"
    lo, hi = s["ci95"]
    return f"{s['mean']:.{k}f} [{lo:.{k}f}, {hi:.{k}f}]"


def report() -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    rows, manifest = [], {"protocol": "docs/PROTOCOL_registration.md", "protocol_sha256": text_sha256(PROTOCOL),
                          "runs": {}}
    for ds, methods in METHODS.items():
        none = os.path.join(RUN_DIR, f"{ds}_{methods[0]}", "records.json")
        none_recs = json.load(open(none, encoding="utf-8"))["records"] if os.path.exists(none) else None
        cand = []
        for m in methods:
            f = os.path.join(RUN_DIR, f"{ds}_{m}", "records.json")
            if not os.path.exists(f):
                continue
            recs = json.load(open(f, encoding="utf-8"))["records"]
            s = summarise(ds, m, recs, none_recs)
            with open(os.path.join(RES_DIR, f"{ds}_{m}.json"), "w", encoding="utf-8") as fh:
                json.dump(s, fh, indent=1, default=float)
            manifest["runs"][f"{ds}_{m}"] = json.load(open(os.path.join(RUN_DIR, f"{ds}_{m}", "run.json"),
                                                           encoding="utf-8"))
            label = m + (f"/{s['frame']}" if s["frame"] else "")
            q1a = s["accuracy"]["q1a_candidate"]["median"] if s["accuracy"]["q1a_candidate"]["n"] else np.inf
            cand.append({"method": label, "eligible": s["eligibility"]["eligible"], "rank_key": s["rank"]["key"],
                         "rank": s["rank"]["value"], "higher": s["rank"]["higher_is_better"], "q1a": q1a})
            fl = s["failure"]
            dn = s["downstream"]
            rows.append({"dataset": ds, "method": label, "pairs": s["n_pairs"],
                         "dropped": sum(s["n_dropped"].values()),
                         "rank metric": s["rank"]["key"], "rank value": fmt(s["rank"]["value"]),
                         "Q1a own (px)": fmt(s["accuracy"]["q1a_candidate"]),
                         "Q1c own patch r": fmt(s["accuracy"].get("q1c_candidate")),
                         "crop coverage own": fmt(s["accuracy"].get("coverage_candidate")),
                         "fail own/fellow/stranger/member": "/".join(
                             "n/a" if fl[k]["rate"] is None else f"{fl[k]['rate']:.3f}"
                             for k in ("own", "fellow", "strangers", "members")),
                         "null c (Q3c)": fmt(s["fairness"]["Q3c_null_c"]),
                         "eligible": s["eligibility"]["eligible"],
                         "biased": s["eligibility"]["biased"],
                         "reasons": "; ".join(s["eligibility"]["reasons"]),
                         "c own P": fmt(dn["c_own_P"]), "c fellow P": fmt(dn["c_fellow_P"]),
                         "c mean P": fmt(dn["c_mean_P"]), "c mean minus own P": fmt(dn["c_mean_minus_own_P"]),
                         "c own N": fmt(dn["c_own_N"])})
        if cand:
            ch = E.choose(cand)
            for r in rows:
                if r["dataset"] == ds:
                    r["recommended"] = r["method"] == ch["recommended"]
            manifest.setdefault("choice", {})[ds] = ch
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES_DIR, "comparison.csv"), index=False)
    with open(os.path.join(RES_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=float)
    lines = ["# Registration study: comparison per data type (aggregates only)", "",
             "Protocol `docs/PROTOCOL_registration.md` (frozen, Amendments 1 to 4). Means with person-cluster bootstrap "
             "95% intervals. Q1a: known-motion error, px on a 224 grid, lower is better. Rank metric: held-out "
             "structure on own same-eye pairs (fundus vessel ring Dice, higher is better; iris centre distance in px, "
             "lower is better), else Q1a. c: calibrated score against matched strangers in pixel space P; a "
             "diagnostic only while the visual QC gate is not met.", ""]
    for ds in METHODS:
        sub = df[df.dataset == ds] if len(df) else df
        if len(sub) == 0:
            continue
        lines += [f"## {ds}", "", f"Choice: {manifest['choice'][ds]}", "", sub.drop(columns="dataset").to_markdown(index=False), ""]
    with open(os.path.join(RES_DIR, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "report"])
    ap.add_argument("--dataset")
    ap.add_argument("--methods")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.dataset)
    elif a.cmd == "run":
        run(a.dataset, a.methods.split(",") if a.methods else METHODS[a.dataset], a.limit)
    else:
        report()


if __name__ == "__main__":
    main()
