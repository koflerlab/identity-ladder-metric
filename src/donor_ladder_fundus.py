"""Donor-averaging ladder on fundus images (docs/PROTOCOL_donor_ladder_fundus.md). Pre-QC; no identity score.

Every image is put in the Amendment 2b disc-fovea frame and oriented disc-right (left eyes reflected), so
same-side and reflected other-side donors share one geometry. Fills are averages of k donor patches; errors are
mean squared differences of field-of-view z-scored green intensity.
Private per-target rows go to the data root; the public aggregate holds counts and summary values only.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import ladder2d as L  # noqa: E402
import ladder2d_register as R  # noqa: E402
import ladder2d_select as S  # noqa: E402

SEED = 20261102
DATA = os.environ.get("DATA_ROOT", r"C:\Projects\data")
PRIVATE = os.path.join(DATA, "projects", "identity-ladder-metric", "donor-ladder-fundus-20261102-v1")
ROOT = os.path.join(os.path.dirname(__file__), "..")
PAPILA = os.path.join(DATA, "datasets", "ophthalmology", "papila", "figshare-14798004-v2", "raw", "PAPILA",
                      "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f")
GRAPE_IMG = os.path.join(DATA, "datasets", "ophthalmology", "grape", "figshare-c6406319-v1", "raw", "CFPs")
G = 256                      # analysis grid
PATCH, RING, N_PATCH = 48, 16, 32
SUPPORT = 0.90               # Amendment 1: 0.95 left no PAPILA window
METRIC = (91, 91, 74)        # 2D-ladder metric box (183-329 at 512) as (y, x, size) on the 256 grid
KS = (1, 2, 4, 8, 16, 32, 64)
DRAWN = ("population", "population_both", "cohort")
COHORTS = DRAWN + ("knn", "fellow_eye", "other_visit")


# ---------- tables ----------

def papila_table() -> pd.DataFrame:
    rows = []
    for eye, side in (("od", "R"), ("os", "L")):
        d = pd.read_excel(os.path.join(PAPILA, "ClinicalData", f"patient_data_{eye}.xlsx"), header=None, skiprows=3)
        d = d.iloc[:, :3].dropna(subset=[0])
        for pid, age, sex in d.itertuples(index=False):
            num = str(pid).strip().lstrip("#").replace("RET", "")
            path = os.path.join(PAPILA, "FundusImages", f"RET{num}{eye.upper()}.jpg")
            rows.append({"person": str(pid).strip(), "side": side, "path": path, "age": age, "sex": sex,
                         "device": "TRC-NW400", "visit": 0, "time": 0.0})
    t = pd.DataFrame(rows)
    return t[t.path.map(os.path.exists)].reset_index(drop=True)


def grape_table() -> pd.DataFrame:
    t = pd.read_csv(S.GRAPE_META)
    t = pd.DataFrame({"person": t["Subject Number"].astype(str), "side": t["Laterality"].map({"OD": "R", "OS": "L"}),
                      "path": t["Corresponding CFP"].astype(str).map(lambda p: os.path.join(GRAPE_IMG, os.path.basename(p))),
                      "age": t["Age"], "sex": t["Gender"], "device": t["Acquisition Device"].astype(str).str.strip(),
                      "visit": t["Visit Number"], "time": t["Interval Years"].astype(float)})
    return t[t.path.map(os.path.exists)].reset_index(drop=True)


# ---------- images ----------

def load(path: str):
    """Disc-right canonical green channel on the 256 grid, z-scored inside the field of view; None if no landmarks."""
    rgb = L.load_base(path, 512, True)
    lm = R.fundus_landmarks(rgb)
    if lm is None:
        return None
    c = R.fundus_canonical(rgb)
    disc_right = lm[0][0] > lm[1][0]
    if not disc_right:
        c = c[:, ::-1]
    g = c[..., 1].reshape(G, 2, G, 2).mean(axis=(1, 3))
    fov = (c[..., 0] > 0.1).reshape(G, 2, G, 2).all(axis=(1, 3))
    v = g[fov]
    g = np.where(fov, (g - v.mean()) / (v.std() + 1e-6), 0.0).astype(np.float32)
    return g, fov, disc_right


def load_all(t: pd.DataFrame, cache: str):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z["img"], z["fov"], z["ok"], z["disc_right"]
    n = len(t)
    img, fov = np.zeros((n, G, G), np.float32), np.zeros((n, G, G), bool)
    ok, dr = np.zeros(n, bool), np.zeros(n, bool)
    for i, p in enumerate(t.path):
        r = load(p)
        if r is not None:
            img[i], fov[i], dr[i] = r
            ok[i] = True
    np.savez_compressed(cache, img=img, fov=fov, ok=ok, disc_right=dr)
    return img, fov, ok, dr


# ---------- design ----------

def patch_positions(fov: np.ndarray, rng) -> list:
    """N_PATCH windows (patch plus ring) with full field-of-view support in >= SUPPORT of images, plus the metric box."""
    w = PATCH + 2 * RING
    ii = np.pad(fov.astype(np.int32).cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    ys = xs = np.arange(0, G - w + 1, 4)
    cand = []
    for y in ys:
        for x in xs:
            s = ii[:, y + w, x + w] - ii[:, y, x + w] - ii[:, y + w, x] + ii[:, y, x]
            if np.mean(s == w * w) >= SUPPORT:
                cand.append((y + RING, x + RING, PATCH))
    pick = rng.choice(len(cand), size=min(N_PATCH, len(cand)), replace=False)
    return [cand[i] for i in sorted(pick)], len(cand)


def ring_mask(size: int) -> np.ndarray:
    m = np.ones((size + 2 * RING, size + 2 * RING), bool)
    m[RING:RING + size, RING:RING + size] = False
    return m


def units(t: pd.DataFrame, ok: np.ndarray, dr: np.ndarray) -> pd.DataFrame:
    """One image per eye (earliest visit) with a valid frame whose disc side agrees with the laterality."""
    t = t.copy()
    t["idx"] = np.arange(len(t))
    t["ok"] = ok
    t["side_conflict"] = ok & (dr != (t.side == "R"))
    good = t[t.ok & ~t.side_conflict].sort_values(["person", "side", "visit"])
    return good.groupby(["person", "side"], as_index=False).first()


def run(name: str, t: pd.DataFrame, out: dict):
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    img, fov, ok, dr = load_all(t, os.path.join(PRIVATE, f"{name}_canonical.npz"))
    u = units(t, ok, dr)
    full = t.assign(idx=np.arange(len(t)), ok=ok, side_conflict=ok & (dr != (t.side == "R")))
    patches, n_cand = patch_positions(fov[u.idx.to_numpy()], rng)
    patches = patches + [METRIC]
    idx = u.idx.to_numpy()
    n = len(u)
    person, side = u.person.to_numpy(), u.side.to_numpy()
    age, sex, dev = u.age.to_numpy(float), u.sex.astype(str).to_numpy(), u.device.to_numpy()

    # Donor pools as index lists into u (other people only).
    def pool(i, both=False, cohort=False):
        m = person != person[i]
        if not both:
            m &= side == side[i]
        if cohort:
            m &= (sex == sex[i]) & (np.abs(age - age[i]) <= 5)
            if name == "GRAPE":
                m &= dev == dev[i]
        return np.flatnonzero(m)

    fellow = np.full(n, -1)
    for i in range(n):
        j = np.flatnonzero((person == person[i]) & (side != side[i]))
        if j.size:
            fellow[i] = j[0]
    # Other visit: same eye, nearest other visit, from the full table (not the one-per-eye units).
    other = np.full(n, -1)
    if name == "GRAPE":
        for i in range(n):
            c = full[(full.person == person[i]) & (full.side == side[i]) & full.ok & ~full.side_conflict
                     & (full.visit != u.visit.iloc[i])]
            if len(c):
                other[i] = c.iloc[np.argmin(np.abs(c.time.to_numpy() - u.time.iloc[i]))].idx

    draws = {}
    for c in DRAWN:
        for i in range(n):
            p = pool(i, both=c == "population_both", cohort=c == "cohort")
            g = np.random.default_rng([SEED, i, DRAWN.index(c)])
            draws[c, i] = g.permutation(p)
    full_pool = {c: {i: draws[c, i] for i in range(n)} for c in DRAWN}

    rows = []  # (target, patch, cohort, k, err, spread)
    for pi, (y, x, s) in enumerate(patches):
        win = (slice(y, y + s), slice(x, x + s))
        X = img[idx][:, win[0], win[1]].reshape(n, -1)
        valid = fov[idx][:, win[0], win[1]].all(axis=(1, 2))
        ry, rx = slice(y - RING, y + s + RING), slice(x - RING, x + s + RING)
        C = img[idx][:, ry, rx][:, ring_mask(s)]
        ring_valid = fov[idx][:, ry, rx][:, ring_mask(s)].all(axis=1)
        sq = (C ** 2).sum(1)
        D = sq[:, None] + sq[None] - 2 * C @ C.T
        Xo = img[:, win[0], win[1]].reshape(len(img), -1)
        vo = fov[:, win[0], win[1]].all(axis=(1, 2))
        for i in range(n):
            if not valid[i]:
                continue
            same = pool(i)
            mu_set = same[valid[same]]
            if mu_set.size < 2:
                continue
            mu = X[mu_set].mean(0)
            xi = X[i]
            for c in DRAWN:
                order = full_pool[c][i]
                order = order[valid[order]]
                for k in KS + ("all",):
                    if k == "all" and c == "population_both":
                        continue
                    sel = order if k == "all" else order[:k]
                    if k != "all" and sel.size < k:
                        continue
                    f = X[sel].mean(0)
                    rows.append((i, pi, c, k, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
            if ring_valid[i]:
                cand = same[valid[same] & ring_valid[same]]
                order = cand[np.argsort(D[i, cand])]
                for k in KS:
                    if order.size >= k:
                        f = X[order[:k]].mean(0)
                        rows.append((i, pi, "knn", k, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
            j = fellow[i]
            if j >= 0 and valid[j]:
                f = X[j]
                rows.append((i, pi, "fellow_eye", 1, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
            j = other[i]
            if j >= 0 and vo[j]:
                f = Xo[j]
                rows.append((i, pi, "other_visit", 1, float(np.mean((xi - f) ** 2)), float(np.mean((f - mu) ** 2))))
        print(f"{name}: patch {pi + 1}/{len(patches)} done, {time.time() - t0:.0f}s", flush=True)

    r = pd.DataFrame(rows, columns=["target", "patch", "cohort", "k", "err", "spread"])
    r["k"] = r["k"].astype(str)
    r["person"] = person[r.target]
    r["region"] = np.where(r.patch == len(patches) - 1, "metric_box", "random_patches")
    r.to_csv(os.path.join(PRIVATE, f"{name}_rows.csv.gz"), index=False)
    out[name] = summarize(r, n_units=n, n_images=len(t), n_ok=int(ok.sum()),
                          n_conflict=int(full.side_conflict.sum()), n_candidates=n_cand,
                          n_fellow=int((fellow >= 0).sum()), n_other=int((other >= 0).sum()))
    return u, img, fov, patches


# ---------- summary ----------

def boot_ratio(tab: pd.DataFrame, col: str, ref: pd.Series, n_boot=2000, seed=SEED):
    """Mean over targets of col / mean over targets of the reference, with a person bootstrap."""
    d = pd.DataFrame({"v": tab[col], "ref": ref.reindex(tab.index)}).dropna()
    persons = d.index.get_level_values("person")
    up = np.unique(persons)
    g = d.groupby(persons).sum()
    cnt = pd.Series(persons).value_counts().reindex(g.index).to_numpy()
    est = d.v.mean() / d.ref.mean()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(up), len(up))
        bs.append(g.v.to_numpy()[s].sum() / g.ref.to_numpy()[s].sum())
    return float(est), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], int(len(d)), int(len(up))


def summarize(r: pd.DataFrame, seed: int = SEED, **counts) -> dict:
    res = {"counts": counts, "regions": {}}
    for region, rr in r.groupby("region"):
        per = rr.groupby(["target", "person", "cohort", "k"])[["err", "spread"]].mean()
        ref = per.xs(("population", "all"), level=("cohort", "k"))["err"]
        reg = {"V_mean_error_of_population_mean": float(ref.mean()), "cohorts": {}}
        for (c, k), tab in per.groupby(level=["cohort", "k"]):
            tab = tab.droplevel(["cohort", "k"])
            e, ci, nt, npers = boot_ratio(tab, "err", ref, seed=seed)
            sp, sci, _, _ = boot_ratio(tab, "spread", ref, seed=seed)
            reg["cohorts"].setdefault(c, {})[k] = {"err_over_V": e, "ci95": ci, "spread_over_V": sp,
                                                   "spread_ci95": sci, "n_targets": nt, "n_persons": npers}
        for c, d in reg["cohorts"].items():
            ks = [k for k in d if k not in ("all",)]
            if len(ks) >= 3:
                kk = np.array([float(k) for k in ks])
                yv = np.array([d[k]["err_over_V"] for k in ks])
                basis = 1 + 1 / kk
                a = float((yv * basis).sum() / (basis ** 2).sum())
                d["fit"] = {"one_minus_R2": a, "R2": 1 - a, "rms_residual": float(np.sqrt(np.mean((yv - a * basis) ** 2)))}
            if c in ("fellow_eye", "other_visit") and "1" in d:
                d["r_implied"] = 1 - d["1"]["err_over_V"] / 2
        if "population" in reg["cohorts"]:
            reg["k_eff_population"] = {k: (1 / v["spread_over_V"] if v["spread_over_V"] > 0 else None)
                                       for k, v in reg["cohorts"]["population"].items() if k.isdigit()}
        res["regions"][region] = reg
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="PAPILA,GRAPE")
    a = ap.parse_args()
    os.makedirs(PRIVATE, exist_ok=True)
    out = {"status": "pre-QC: disc-fovea landmarks not visually accepted", "protocol": "docs/PROTOCOL_donor_ladder_fundus.md",
           "seed": SEED, "grid": G, "patch": PATCH, "ring": RING, "ks": list(KS)}
    for name in a.datasets.split(","):
        t = papila_table() if name == "PAPILA" else grape_table()
        run(name, t, out)
    shutil.copy2(__file__, os.path.join(PRIVATE, "code_snapshot.py"))
    with open(os.path.join(ROOT, "results", "donor_ladder_fundus.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    print(json.dumps(out["PAPILA"]["counts"] if "PAPILA" in out else {}, indent=1))


if __name__ == "__main__":
    main()
