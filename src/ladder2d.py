"""The identity ladder on real 2D images: GRAPE fundus and bilateral iris (docs/PROTOCOL_2d_ladder.md, frozen).

For every selected pair (anchor x, candidate y) and every representation, the distance from x to the candidate slot
is calibrated against the pair's nuisance-matched strangers: c = P(d(x, s) > d) + half the ties. The candidate slot
holds the person's own image (lambda = 1), points of the shrinkage path y_lambda = mu + lambda (y - mu), and the
population mean mu (lambda = 0) of M matched strangers disjoint from the reference.

Refuses to run if the protocol's input hashes differ. Writes aggregates to results/ladder2d.json and provenance to
results/ladder2d_manifest.json; per-pair scores and per-file image hashes stay in the private run folder.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import ladder2d_select as sel  # noqa: E402
import reid  # noqa: E402
from grape_calibration import calibrated_score  # noqa: E402
from run_grape import EXT  # noqa: E402

PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PROTOCOL = os.path.join(PROJECT, "docs", "PROTOCOL_2d_ladder.md")
GRAPE_IMG = os.path.join(sel.DATA_ROOT, "datasets", "ophthalmology", "grape", "figshare-c6406319-v1", "raw", "CFPs")
RETFOUND = os.path.join(EXT, "features", "GRAPE")
MODEL_DIR = os.path.join(sel.DATA_ROOT, "projects", "identity-ladder-metric", "models", "dinov2-small-ed25f3a")
EXPECTED = {  # from the frozen protocol
    "grape_metadata": (sel.GRAPE_META, "bc79566f4687f7512628ee26959eca0fdb11dd859d217ebf82be99035fd2e02e"),
    "selection_grape": (os.path.join(sel.OUT_DIR, "selection_grape.json"),
                        "d3c236d60d125f886ea93ed2804a815b0007e12ccc1082c01b36c2d1a7a4708a"),
    "selection_iris": (os.path.join(sel.OUT_DIR, "selection_iris.json"),
                       "c3a41f09ac7d622c8215788b99c2d67ce1ecfc729d05ba33aab1b319240e107c"),
    "dinov2_weights": (os.path.join(MODEL_DIR, "model.safetensors"),
                       "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"),
    "dinov2_config": (os.path.join(MODEL_DIR, "config.json"),
                      "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"),
}
SEED = sel.SEED
N_BOOT = 2000
LAMBDAS = (1.0, 0.75, 0.5, 0.25, 0.0)
TAUS = (0.5, 0.9, 0.95, 0.99)
BASE, P_SIZE, V_SIZE = 224, 64, 512
V_FRACTIONS = (0.12, 0.08, 0.16)  # primary, then sensitivity S5
THREADS = 4


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def guard() -> dict:
    found = {k: sha256(p) for k, (p, _) in EXPECTED.items()}
    bad = [k for k, (_, want) in EXPECTED.items() if found[k] != want]
    if bad:
        raise SystemExit(f"input hash differs from the frozen protocol: {bad}")
    return found


# ---------- images and representations ----------

def load_base(path: str, size: int, rgb: bool) -> np.ndarray:
    """Centre square crop, BOX resize, [0, 1]; RGB (H, W, 3) or grey (H, W) by ITU-R 601 luma."""
    im = Image.open(path)
    im = im.convert("RGB" if rgb else "L")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BOX)
    return np.asarray(im, dtype=np.float32) / 255.0


def grey(b: np.ndarray) -> np.ndarray:
    return b if b.ndim == 2 else b @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def box_resize(a: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((size, size), Image.BOX), np.float64)


def rep_pixels_raw(b: np.ndarray, size: int = P_SIZE) -> np.ndarray:
    return box_resize(grey(b), size).ravel()


def zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def rep_nuisance(b: np.ndarray) -> np.ndarray:
    g = grey(b)
    return np.array([g.mean(), g.std()], dtype=np.float64)


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class Dino:
    def __init__(self, model_dir: str = MODEL_DIR):
        import torch
        from transformers import Dinov2Model
        torch.set_num_threads(THREADS)
        self.torch = torch
        self.model = Dinov2Model.from_pretrained(model_dir, local_files_only=True).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, bases: list) -> np.ndarray:
        out = []
        for i in range(0, len(bases), 32):
            x = np.stack([b if b.ndim == 3 else np.repeat(b[..., None], 3, -1) for b in bases[i:i + 32]])
            t = (self.torch.from_numpy(x).permute(0, 3, 1, 2).float() - self.mean) / self.std
            with self.torch.no_grad():
                out.append(self.model(pixel_values=t).pooler_output.numpy().astype(np.float64))
        return np.concatenate(out)


# ---------- vessel label space (GRAPE) ----------

def vessel_masks(rgb512: np.ndarray, fractions=V_FRACTIONS, min_size: int = 30) -> list:
    """Classical vessel segmentation at each top-fraction threshold of Frangi vesselness inside the field of view."""
    from skimage.exposure import equalize_adapthist
    from skimage.filters import frangi
    from skimage.morphology import remove_small_objects
    fov = rgb512[..., 0] > 0.1
    g = equalize_adapthist(np.clip(rgb512[..., 1], 0, 1))
    v = frangi(g, sigmas=range(1, 5), black_ridges=True)
    v = np.where(fov, v, 0)
    out = []
    for f in fractions:
        inside = v[fov]
        thr = np.quantile(inside, 1 - f) if inside.size else np.inf
        out.append(remove_small_objects((v > thr) & fov, max_size=min_size - 1))
    return out


def topology(mask: np.ndarray) -> tuple:
    """(connected components, branch points) of the 8-connected skeleton."""
    from scipy.ndimage import convolve, label
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    n_comp = int(label(sk, structure=np.ones((3, 3)))[1])
    nb = convolve(sk.astype(np.int32), np.ones((3, 3), np.int32), mode="constant") - sk
    return n_comp, int(((nb >= 3) & sk).sum())


def dice_distance(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(1 - 2 * (a & b).sum() / s) if s else 0.0


# ---------- statistics ----------

def person_weights(people: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """Bootstrap count of every person in every resample (person-cluster bootstrap)."""
    ids, idx = np.unique(people, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n_boot)
    return idx, counts


def boot_mean(values: np.ndarray, idx: np.ndarray, counts: np.ndarray) -> dict:
    """Mean with a percentile 95% interval; values of one rung, idx their person index into counts."""
    w = counts[:, idx].astype(float)
    ok = w.sum(1) > 0
    stats = (w[ok] @ values) / w[ok].sum(1)
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]}


def boot_diff(va, ia, vb, ib, counts) -> dict:
    """Difference of two rung means under the same person resamples (unpaired rungs, shared people)."""
    wa, wb = counts[:, ia].astype(float), counts[:, ib].astype(float)
    ok = (wa.sum(1) > 0) & (wb.sum(1) > 0)
    stats = (wa[ok] @ va) / wa[ok].sum(1) - (wb[ok] @ vb) / wb[ok].sum(1)
    return {"diff": float(va.mean() - vb.mean()),
            "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]}


def describe(values: np.ndarray, idx: np.ndarray, counts: np.ndarray) -> dict:
    return {**boot_mean(values, idx, counts), "median": float(np.median(values)), "n_pairs": int(len(values)),
            "tolerance": {f"c>={t}": float((values >= t).mean()) for t in TAUS}}


def label_mean(ci) -> str:
    lo, hi = ci
    return "rewards the mean" if lo > 0 else "places the mean below the person" if hi < 0 else "not separated"


# ---------- the run ----------

class Images:
    """Base images of one dataset, loaded once; candidate-slot images built on demand."""

    def __init__(self, paths: dict, rgb: bool, v: bool):
        self.paths, self.rgb, self.v = paths, rgb, v
        self.b, self.b512, self.hashes = {}, {}, {}
        for name, p in paths.items():
            self.hashes[name] = sha256(p)
            self.b[name] = load_base(p, BASE, rgb)
            if v:
                self.b512[name] = (load_base(p, V_SIZE, True) * 255).astype(np.uint8)

    def mean(self, names: list, big: bool = False) -> np.ndarray:
        src = self.b512 if big else self.b
        return np.mean([src[n].astype(np.float32) / (255.0 if big else 1.0) for n in names], axis=0)

    def get(self, name: str, big: bool = False) -> np.ndarray:
        return self.b512[name].astype(np.float32) / 255.0 if big else self.b[name]


def score_dataset(ds: str, pairs: list, imgs: Images, dino: Dino, retfound: dict | None, rng_s4,
                  u_s4: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (pair, representation, slot, variant) with distance d and calibrated c."""
    names = sorted(imgs.b)
    nz = np.array([rep_nuisance(imgs.b[n]) for n in names])
    n_mu, n_sd = nz.mean(0), nz.std(0)
    feat = {"P": lambda b: zscore(rep_pixels_raw(b)), "P32": lambda b: zscore(rep_pixels_raw(b, 32)),
            "P16": lambda b: zscore(rep_pixels_raw(b, 16)), "N": lambda b: (rep_nuisance(b) - n_mu) / n_sd}
    dist = {"P": euclid, "P32": euclid, "P16": euclid, "N": euclid, "D": cosine_distance, "R": euclid}
    real_d = dict(zip(names, dino([imgs.b[n] for n in names])))
    real_f = {k: {n: f(imgs.b[n]) for n in names} for k, f in feat.items()}
    vcache = {}
    if u_s4 is not None:
        keys = tuple(k for k in sel.CFG["grape" if ds == "GRAPE" else "iris"]["keys"] if k != "side")
        s4_pool = sel.StrangerPool(u_s4[u_s4.image.isin(imgs.b)], keys)
        s4_rows = u_s4.set_index("image")

    def vmasks(key, rgb512):
        if key not in vcache:
            vcache[key] = vessel_masks(rgb512)
        return vcache[key]

    rows = []
    for p in pairs:
        x, y, ref, mem = p["anchor"], p["candidate"], p["reference"], p["mean_members"]
        mu, yb = imgs.mean(mem), imgs.get(y)
        slot = {lam: mu + lam * (yb - mu) for lam in LAMBDAS}  # lambda = 1 is y itself
        slot_f = {k: {lam: (real_f[k][y] if lam == 1 else f(slot[lam])) for lam in LAMBDAS} for k, f in feat.items()}
        path = [lam for lam in LAMBDAS if lam != 1]
        slot_f["D"] = {1.0: real_d[y], **dict(zip(path, dino([slot[lam] for lam in path])))}
        anchor_f = {k: real_f[k][x] for k in feat} | {"D": real_d[x]}
        ref_f = {k: [real_f[k][s] for s in ref] for k in feat} | {"D": [real_d[s] for s in ref]}
        halves = (ref[: len(ref) // 2], ref[len(ref) // 2:])
        base = {"dataset": ds, "rung": p["rung"], "person": p["person"], "pair": f"{p['anchor']}|{p['candidate']}"}

        def emit(rep, variant, slot_d, ref_d):
            ref_d = np.asarray(ref_d)
            for lam, d in slot_d.items():
                rows.append({**base, "rep": rep, "variant": variant, "lam": lam, "d": d,
                             "c": float(calibrated_score(np.array([d]), ref_d)[0]), "k": len(ref_d)})

        for k in ("P", "N", "D", "P32", "P16"):
            ref_d = [dist[k](anchor_f[k], r) for r in ref_f[k]]
            lams = LAMBDAS if k in ("P", "N", "D") else (1.0, 0.0)
            emit(k, "primary" if k in ("P", "N", "D") else "S2_" + k, {l: dist[k](anchor_f[k], slot_f[k][l]) for l in lams},
                 ref_d)
            if k in ("P", "N", "D"):
                for h, half in zip("AB", halves):
                    hd = [dist[k](anchor_f[k], real_f[k][s] if k != "D" else real_d[s]) for s in half]
                    emit(k, f"S3_half_{h}", {1.0: dist[k](anchor_f[k], slot_f[k][1.0]),
                                             0.0: dist[k](anchor_f[k], slot_f[k][0.0])}, hd)
        if p["rung"].startswith("fellow"):  # S1: candidate slot flipped left-right
            fl = {l: zscore(rep_pixels_raw(slot[l][:, ::-1])) for l in (1.0, 0.0)}
            emit("P", "S1_flipped", {l: euclid(anchor_f["P"], fl[l]) for l in fl},
                 [euclid(anchor_f["P"], zscore(rep_pixels_raw(imgs.b[s][:, ::-1]))) for s in ref])
        if u_s4 is not None:  # S4: reference matched on camera (or dataset and sensor) only, not on eye side
            s4 = s4_pool.draw(s4_rows.loc[y], {p["person"]}, len(ref), 1, rng_s4)
            emit("P", "S4_side_pooled", {1.0: dist["P"](anchor_f["P"], slot_f["P"][1.0]),
                                         0.0: dist["P"](anchor_f["P"], slot_f["P"][0.0])},
                 [euclid(anchor_f["P"], real_f["P"][s]) for s in s4])
        if retfound is not None:
            emit("R", "primary", {1.0: euclid(retfound[x], retfound[y])}, [euclid(retfound[x], retfound[s]) for s in ref])
        if imgs.v:
            mx = vmasks(x, imgs.get(x, True))
            mu5, y5 = imgs.mean(mem, True), imgs.get(y, True)
            ms = {l: vmasks(y, y5) if l == 1 else vessel_masks(mu5 + l * (y5 - mu5)) for l in LAMBDAS}
            mr = [vmasks(s, imgs.get(s, True)) for s in ref]
            for fi, frac in enumerate(V_FRACTIONS):
                variant = "primary" if fi == 0 else f"S5_top{int(frac * 100)}"
                tx = topology(mx[fi])
                tr = [topology(m[fi]) for m in mr]
                emit("V_dice", variant, {l: dice_distance(mx[fi], ms[l][fi]) for l in LAMBDAS},
                     [dice_distance(mx[fi], m[fi]) for m in mr])
                if fi == 0:
                    ts = {l: topology(ms[l][0]) for l in LAMBDAS}
                    for j, nm in enumerate(("V_components", "V_branches")):
                        emit(nm, "primary", {l: abs(tx[j] - ts[l][j]) for l in LAMBDAS}, [abs(tx[j] - t[j]) for t in tr])
            if p["rung"].startswith("fellow"):
                emit("V_dice", "S1_flipped", {1.0: dice_distance(mx[0], ms[1.0][0][:, ::-1]),
                                              0.0: dice_distance(mx[0], ms[0.0][0][:, ::-1])},
                     [dice_distance(mx[0], m[0][:, ::-1]) for m in mr])
        # path diagnostics on 64 x 64 grey before z-scoring, where the path is linear
        xa, ma, ya = rep_pixels_raw(imgs.b[x]), rep_pixels_raw(mu), rep_pixels_raw(yb)
        for lam in LAMBDAS:
            yl = ma + lam * (ya - ma)
            r, t = yl - ma, xa - ma
            rows.append({**base, "rep": "P_raw_path", "variant": "diagnostic", "lam": lam,
                         "d_x_y": euclid(xa, yl), "d_y_mu": euclid(yl, ma), "d_x_mu": euclid(xa, ma),
                         "residual_ratio": float(np.linalg.norm(r) / max(np.linalg.norm(ya - ma), 1e-12)),
                         "target_projection": float(r @ t / max(t @ t, 1e-12)),
                         "residual_cosine": float(r @ t / max(np.linalg.norm(r) * np.linalg.norm(t), 1e-12))})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> dict:
    out = {}
    sc = df[df.variant != "diagnostic"]
    for ds, g in sc.groupby("dataset"):
        _, counts = person_weights(g.person.unique())
        pid = {p: i for i, p in enumerate(np.unique(g.person.unique()))}
        res = {"rungs": {}, "contrasts": {}, "path_diagnostics": {}}
        for (rep, variant, rung), h in g.groupby(["rep", "variant", "rung"]):
            e = res["rungs"].setdefault(f"{rep} | {variant}", {}).setdefault(rung, {"people": int(h.person.nunique())})
            for lam, hl in h.groupby("lam"):
                e[f"lambda={lam:g}"] = describe(hl.c.to_numpy(), hl.person.map(pid).to_numpy(), counts)
        prim = g[g.variant == "primary"]
        own = prim[prim.lam == 1.0].pivot_table(index=["rung", "pair", "person"], columns="rep", values="c").reset_index()
        mean = prim[prim.lam == 0.0].pivot_table(index=["rung", "pair", "person"], columns="rep", values="c").reset_index()
        for rung, o in own.groupby("rung"):
            ix = o.person.map(pid).to_numpy()
            c = res["contrasts"].setdefault(rung, {})
            for rep in ("P", "D"):
                c[f"{rep} minus N (own)"] = boot_mean((o[rep] - o["N"]).to_numpy(), ix, counts)
            m = mean[mean.rung == rung].set_index("pair").loc[o.pair]
            for rep in [r for r in ("P", "D", "N", "V_dice", "V_components", "V_branches") if r in o]:
                c[f"{rep} mean minus own"] = boot_mean((m[rep].to_numpy() - o[rep].to_numpy()), ix, counts)
        rungs = sorted(own.rung.unique())

        def diff(a, b, rep="P"):
            oa, ob = own[own.rung == a], own[own.rung == b]
            return boot_diff(oa[rep].to_numpy(), oa.person.map(pid).to_numpy(), ob[rep].to_numpy(),
                             ob.person.map(pid).to_numpy(), counts)

        pairs_c = ([("same_camera_<1y", "same_camera_1y+"), ("same_camera_<1y", "other_camera"),
                    ("fellow_eye_same_visit", "same_camera_<1y")] if ds == "GRAPE"
                   else [("fellow_eye_same_sitting", "same_eye_same_sitting")])
        res["rung_differences"] = {f"{a} minus {b} | {rep}": diff(a, b, rep) for a, b in pairs_c for rep in ("P", "D", "N")
                                   if a in rungs and b in rungs}
        dg = df[(df.dataset == ds) & (df.variant == "diagnostic")]
        for lam, h in dg.groupby("lam"):
            res["path_diagnostics"][f"lambda={lam:g}"] = {k: float(h[k].mean()) for k in
                                                         ("d_x_y", "d_y_mu", "d_x_mu", "residual_ratio",
                                                          "target_projection", "residual_cosine")}
        out[ds] = res
    return out


def interpret(agg: dict) -> dict:
    """The prespecified labels, applied mechanically."""
    lab = {}
    for ds, r in agg.items():
        L = lab.setdefault(ds, {})
        for rung, c in r["contrasts"].items():
            for rep in ("P", "D"):
                key = f"{rep} | primary"
                if key not in r["rungs"] or rung not in r["rungs"][key]:
                    continue
                own = r["rungs"][key][rung]["lambda=1"]["ci95"]
                L[f"{rung} | {rep} identity persists"] = bool(own[0] > 0.5 and c[f"{rep} minus N (own)"]["ci95"][0] > 0)
                L[f"{rung} | {rep} mean"] = label_mean(c[f"{rep} mean minus own"]["ci95"])
            if "V_dice mean minus own" in c:
                L[f"{rung} | V_dice mean"] = label_mean(c["V_dice mean minus own"]["ci95"])
            n = r["rungs"]["N | primary"][rung]["lambda=1"]["ci95"]
            L[f"{rung} | nuisance-driven"] = bool(n[0] > 0.5)
        for name, d in r["rung_differences"].items():
            if not name.startswith("fellow"):  # C2 only; the protocol gives the C3 (bilateral) contrast no label
                L[name] = "lowers the score" if d["ci95"][0] > 0 else "no detectable effect"
        if ds == "GRAPE" and "same_camera_1y+" in r["contrasts"]:
            p_lab, v_lab = L.get("same_camera_1y+ | P mean"), L.get("same_camera_1y+ | V_dice mean")
            L["label space penalises the mean (primary rung)"] = bool(
                v_lab == "places the mean below the person" and p_lab in ("rewards the mean", "not separated"))
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first n pairs per dataset (timing only; no results written)")
    ap.add_argument("--from-scores", action="store_true",
                    help="re-aggregate the saved private pair_scores.csv (no image read, no score recomputed)")
    a = ap.parse_args()
    if a.from_scores:
        guard()
        df = pd.read_csv(os.path.join(sel.OUT_DIR, "pair_scores.csv"), low_memory=False)
        write_results(df, note="re-aggregated from the saved pair scores")
        return
    t0 = time.time()
    hashes = guard()
    dino = Dino()
    grape_sel = json.load(open(EXPECTED["selection_grape"][0], encoding="utf-8"))
    iris_sel = json.load(open(EXPECTED["selection_iris"][0], encoding="utf-8"))
    u_grape = sel.grape_universe(pd.read_csv(sel.GRAPE_META))
    utiris, _ = sel.utiris_table()
    u_iris = sel.iris_universe(sel.upol_table(), utiris)
    retfound = reid.load_features_safe(RETFOUND)
    frames, image_hashes = [], {}
    jobs = [("GRAPE", grape_sel, u_grape, {n: os.path.join(GRAPE_IMG, n) for n in u_grape.image}, True, True, retfound)]
    for ds, root in (("UPOL", sel.UPOL), ("UTIRIS", sel.UTIRIS)):
        u = u_iris[u_iris.dataset == ds]
        jobs.append((ds, iris_sel, u, {n: os.path.join(root, n) for n in u.image}, False, False, None))
    for ds, s, u, paths, rgb, v, rf in jobs:
        prs = [p for p in s["pairs"] if p["dataset"] == ds]
        if a.limit:
            prs = prs[: a.limit]
        need = sorted({n for p in prs for n in [p["anchor"], p["candidate"], *p["reference"], *p["mean_members"]]})
        imgs = Images({n: paths[n] for n in (sorted(paths) if not a.limit else need)}, rgb, v)
        image_hashes[ds] = imgs.hashes
        t1 = time.time()
        frames.append(score_dataset(ds, prs, imgs, dino, rf, np.random.default_rng([SEED, 4]), u))
        print(f"{ds}: {len(prs)} pairs, {len(imgs.b)} images, {time.time() - t1:.0f} s", flush=True)
    if a.limit:
        print(f"timing run only, total {time.time() - t0:.0f} s")
        return
    df = pd.concat(frames, ignore_index=True)
    private = sel.OUT_DIR
    df.to_csv(os.path.join(private, "pair_scores.csv"), index=False)
    with open(os.path.join(private, "image_sha256.json"), "w", encoding="utf-8") as fh:
        json.dump(image_hashes, fh, indent=0, sort_keys=True)
    write_results(df)
    write_manifest(hashes, image_hashes, t0)


def write_results(df: pd.DataFrame, note: str = "") -> None:
    agg = aggregate(df)
    out = {"protocol": "docs/PROTOCOL_2d_ladder.md (frozen)", "seed": SEED, "n_boot": N_BOOT,
           "note": "c = P(d_stranger > d) + half ties against the pair's nuisance-matched strangers; lambda=1 own image, "
                   "lambda=0 population mean of 16 matched strangers; intervals are person-cluster bootstrap 95%",
           "aggregation": note or "computed in the scoring run",
           "results": agg, "prespecified_labels": interpret(agg)}
    with open(os.path.join(PROJECT, "results", "ladder2d.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["prespecified_labels"], indent=1))


def write_manifest(hashes: dict, image_hashes: dict, t0: float) -> None:
    import skimage
    import torch
    import transformers
    commit = subprocess.run(["git", "-C", PROJECT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"protocol_sha256": sha256(PROTOCOL), "code_sha256": {f: sha256(os.path.join(PROJECT, "src", f))
                for f in ("ladder2d.py", "ladder2d_select.py", "grape_calibration.py", "iris_calibration.py")},
                "inputs_sha256": hashes, "retfound_dir_sha256": {f: sha256(os.path.join(RETFOUND, f))
                                                                for f in sorted(os.listdir(RETFOUND)) if f.endswith(".pkl")},
                "image_set_sha256": {ds: hashlib.sha256("\n".join(f"{n}:{h}" for n, h in sorted(hs.items())).encode())
                                     .hexdigest() for ds, hs in image_hashes.items()},
                "n_images": {ds: len(hs) for ds, hs in image_hashes.items()},
                "git_commit": commit, "seed": SEED, "threads": THREADS, "run_seconds": round(time.time() - t0),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit-image": skimage.__version__, "torch": torch.__version__,
                             "transformers": transformers.__version__, "Pillow": Image.__version__},
                "private_outputs": "data root, projects/identity-ladder-metric/run-2026-11-02-2d-ladder"}
    with open(os.path.join(PROJECT, "results", "ladder2d_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
