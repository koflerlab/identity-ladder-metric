"""Aggregates and decisions of docs/PROTOCOL_registration.md from per-pair records (Q1 to Q4, R1 to R5).

A per-pair record is a dict written by reg_run.py (private). Everything here returns aggregates only. Intervals:
person-cluster bootstrap, 2,000 resamples, seed 20261101 (ladder2d.person_weights).
"""
import numpy as np

import ladder2d as L

FAIL_MAX = 0.10           # failure in every role
FAIL_GAP = 0.05           # Q3a: interval entirely above 5 points against the person
GEO_GAP = 1.0             # Q3b: interval entirely above 1 px (224 grid) in the person's favour
DOF = {"G0": 0, "I0": 0, "G1": 1, "H1": 1, "H2": 1, "I1": 1, "I2": 1, "I3": 1,
       "G2": 2, "G3": 2, "I4": 2, "I5": 2, "G4": 4}
HELD_OUT = {"G0": ("disc", "fovea"), "G2": ("disc", "fovea"), "G3": ("disc", "fovea"), "G4": ("disc", "fovea"),
            "G1": (), "H1": (), "H2": (), "I0": ("pupil", "limbus"), "I1": ("pupil",), "I2": ("limbus",),
            "I3": (), "I4": ()}
CLOSEST = {"GRAPE": "same_camera_<1y", "UTIRIS": "same_eye_same_sitting", "UPOL": "same_eye_same_sitting"}


def held_out(method: str, frame: str = "") -> tuple:
    """I5 and iris H1 inherit the held-out structure of their frame (I1 or I2)."""
    if frame:
        return HELD_OUT[frame]
    return HELD_OUT[method]


def fellow(r: dict) -> bool:
    return r["rung"].startswith("fellow")


def boot(values, people, counts_cache: dict) -> dict:
    """Mean with the person-cluster interval over the pairs that have a finite value."""
    v = np.asarray(values, float)
    p = np.asarray(people)
    keep = np.isfinite(v)
    if keep.sum() == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    v, p = v[keep], p[keep]
    key = tuple(p)
    if key not in counts_cache:
        counts_cache[key] = L.person_weights(p)
    idx, counts = counts_cache[key]
    out = L.boot_mean(v, idx, counts)
    return {**out, "median": float(np.median(v)), "n": int(len(v))}


def boot_unpaired(va, pa, vb, pb) -> dict:
    """Difference of two group means under shared person resamples (ladder2d.boot_diff)."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    pa, pb = np.asarray(pa), np.asarray(pb)
    ka, kb = np.isfinite(va), np.isfinite(vb)
    va, pa, vb, pb = va[ka], pa[ka], vb[kb], pb[kb]
    if len(va) == 0 or len(vb) == 0:
        return {"diff": None, "ci95": [None, None]}
    ids = np.unique(np.concatenate([pa, pb]))
    pos = {x: i for i, x in enumerate(ids)}
    _, counts = L.person_weights(ids)
    return L.boot_diff(va, np.array([pos[x] for x in pa]), vb, np.array([pos[x] for x in pb]), counts)


def failure_rates(recs: list) -> dict:
    """Per role: failures / attempts (pooled), and the per-pair values used for the parity tests."""
    out = {}
    anchor = [r["fail"]["anchor"] for r in recs]
    out["anchor"] = {"fail": int(sum(anchor)), "attempts": len(anchor)}
    att = [r for r in recs if not r["fail"]["anchor"]]
    for role, sel in (("own", lambda r: not fellow(r)), ("fellow", fellow)):
        xs = [r["fail"]["candidate"] for r in att if sel(r)]
        out[role] = {"fail": int(sum(xs)), "attempts": len(xs)}
    for role in ("strangers", "members"):
        f = sum(r["fail"][role][0] for r in att)
        n = sum(r["fail"][role][1] for r in att)
        out[role] = {"fail": int(f), "attempts": int(n)}
    for k, v in out.items():
        v["rate"] = v["fail"] / v["attempts"] if v["attempts"] else None
    return out


def _rate(r, role):
    f, n = r["fail"][role]
    return f / n if n else np.nan


def fairness(recs: list, method: str, frame: str = "") -> dict:
    """Q3a to Q3e. Differences are 'other role minus own'; for errors a positive value favours the person."""
    att = [r for r in recs if not r["fail"]["anchor"]]
    same = [r for r in att if not fellow(r)]
    fel = [r for r in att if fellow(r)]
    pe = lambda rs: [r["person"] for r in rs]  # noqa: E731
    cc = {}
    q = {}
    q["Q3a_strangers_minus_own"] = boot([_rate(r, "strangers") - r["fail"]["candidate"] for r in same], pe(same), cc)
    q["Q3a_members_minus_own"] = boot([_rate(r, "members") - r["fail"]["candidate"] for r in same], pe(same), cc)
    q["Q3a_fellow_minus_own"] = boot_unpaired([r["fail"]["candidate"] for r in fel], pe(fel),
                                              [r["fail"]["candidate"] for r in same], pe(same))
    ok = [r for r in att if not r["dropped"]]
    s_ok = [r for r in ok if not fellow(r)]
    f_ok = [r for r in ok if fellow(r)]
    geo = ("q1a",) + tuple(held_out(method, frame))
    for k in geo:
        get = (lambda r, role: r["q1a"][role]) if k == "q1a" else (lambda r, role: r["q1b"].get(role, {}).get(k, np.nan))
        q[f"Q3b_{k}_strangers_minus_own"] = boot([get(r, "strangers") - get(r, "candidate") for r in s_ok],
                                                 pe(s_ok), cc)
        q[f"Q3b_{k}_fellow_minus_own"] = boot_unpaired([get(r, "candidate") for r in f_ok], pe(f_ok),
                                                       [get(r, "candidate") for r in s_ok], pe(s_ok))
    q["Q3c_null_c"] = boot([r["c"]["null_P"] for r in ok], pe(ok), cc)
    q["Q3d_swap_minus_own"] = boot([r["c"]["swap_P"] - r["c"]["own_P"] for r in ok], pe(ok), cc)
    q["Q3e_reflect_minus_own"] = boot([r["c"]["reflect_P"] - r["c"]["own_P"] for r in s_ok], pe(s_ok), cc)
    q["vessel_dice_own_minus_strangers"] = boot(  # descriptive only (Amendment 2)
        [r["q1b"].get("candidate", {}).get("dice", np.nan) - r["q1b"].get("strangers", {}).get("dice", np.nan) for r in s_ok],
        pe(s_ok), cc)
    return q


def _lo(ci):
    return ci[0] if ci and ci[0] is not None else None


def eligibility(fail: dict, fair: dict, method: str, frame: str = "", jac_nonpos: float | None = None) -> dict:
    """Prespecified rule; returns {'eligible': bool, 'biased': bool, 'reasons': [...]}."""
    reasons, biased = [], False
    for role, v in fail.items():
        if v["rate"] is not None and v["rate"] > FAIL_MAX:
            reasons.append(f"failure {role} {v['rate']:.3f} > {FAIL_MAX}")
    for k in ("Q3a_strangers_minus_own", "Q3a_members_minus_own", "Q3a_fellow_minus_own"):
        lo = _lo(fair[k]["ci95"])
        if lo is not None and lo > FAIL_GAP:
            reasons.append(f"{k} lower bound {lo:.3f} > {FAIL_GAP}")
            biased = True
    for k in ("q1a",) + tuple(held_out(method, frame)):
        for side in ("strangers", "fellow"):
            lo = _lo(fair[f"Q3b_{k}_{side}_minus_own"]["ci95"])
            if lo is not None and lo > GEO_GAP:
                reasons.append(f"Q3b {k} {side} minus own lower bound {lo:.2f} px > {GEO_GAP}")
                biased = True
    ci = fair["Q3c_null_c"]["ci95"]
    if ci[0] is not None and not (ci[0] <= 0.5 <= ci[1]):
        reasons.append(f"Q3c null c interval {ci} excludes 0.5")
        biased = True
    for k in ("Q3d_swap_minus_own", "Q3e_reflect_minus_own"):
        ci = fair[k]["ci95"]
        if ci[0] is not None and not (ci[0] <= 0 <= ci[1]):
            reasons.append(f"{k} interval {ci} excludes 0")
            biased = True
    if jac_nonpos is not None and jac_nonpos > 0:
        reasons.append("folding")
    return {"eligible": not reasons, "biased": biased, "reasons": reasons}


def accuracy(recs: list, method: str, frame: str = "") -> dict:
    """Q1 summaries per role (own same-eye pairs for ranking)."""
    ok = [r for r in recs if not r["fail"]["anchor"] and not r["dropped"]]
    same = [r for r in ok if not fellow(r)]
    pe = lambda rs: [r["person"] for r in rs]  # noqa: E731
    cc = {}
    out = {}
    for role in ("candidate", "strangers", "members", "null"):
        rs = same if role != "candidate" else same
        out[f"q1a_{role}"] = boot([r["q1a"].get(role, np.nan) for r in rs], pe(rs), cc)
        for k in ("dice", "disc", "fovea", "pupil", "limbus"):
            vals = [r["q1b"].get(role, {}).get(k, np.nan) for r in rs]
            if np.isfinite(np.asarray(vals, float)).any():
                out[f"q1b_{k}_{role}"] = boot(vals, pe(rs), cc)
        for k in ("q1c", "coverage", "q1d"):
            vals = [r[k].get(role, np.nan) for r in rs]
            if np.isfinite(np.asarray(vals, float)).any():
                out[f"{k}_{role}"] = boot(vals, pe(rs), cc)
        lt = [r["coverage_lt95"].get(role, np.nan) for r in rs]
        if np.isfinite(np.asarray(lt, float)).any():
            out[f"coverage_lt95_share_{role}"] = boot(lt, pe(rs), cc)
    fel = [r for r in ok if fellow(r)]
    for k in ("q1c", "coverage"):
        vals = [r[k].get("candidate", np.nan) for r in fel]
        if np.isfinite(np.asarray(vals, float)).any():
            out[f"{k}_fellow"] = boot(vals, pe(fel), cc)
    if method == "G4":
        out["q1e_jac_p99_abslog_candidate"] = boot([r["q1e"].get("candidate", np.nan) for r in same], pe(same), cc)
    return out


def ranking_value(acc: dict, dataset: str, method: str, frame: str = "") -> tuple:
    """(key, summary, higher_is_better) for R4: held-out Q1b on own same-eye pairs, else Q1a."""
    if dataset == "GRAPE" and "q1b_dice_candidate" in acc:
        return "q1b_dice_candidate", acc["q1b_dice_candidate"], True
    for k in held_out(method, frame):
        if f"q1b_{k}_candidate" in acc:
            return f"q1b_{k}_candidate", acc[f"q1b_{k}_candidate"], False
    return "q1a_candidate", acc["q1a_candidate"], False


def choose(rows: list) -> dict:
    """R4 among eligible rows (dicts with method, eligible, rank_key, rank, higher, q1a). Overlapping intervals of the
    best are tied; the tie goes to fewer degrees of freedom, then lower median Q1a."""
    el = [r for r in rows if r["eligible"] and r["rank"]["mean"] is not None]
    if not el:
        return {"recommended": None, "verdict": "not registrable under this protocol", "tied": []}
    # rank only within the same ranking key as the best-covered key (Q1b before Q1a)
    keys = [r["rank_key"] for r in el]
    pref = next((k for k in keys if k.startswith("q1b")), keys[0])
    el = [r for r in el if r["rank_key"] == pref]
    sign = 1 if el[0]["higher"] else -1
    best = max(el, key=lambda r: sign * r["rank"]["mean"])
    blo, bhi = best["rank"]["ci95"]
    tied = [r for r in el if not (r["rank"]["ci95"][1] < blo or r["rank"]["ci95"][0] > bhi)]
    pick = min(tied, key=lambda r: (DOF[r["method"].split("/")[0]], r["q1a"]))
    return {"recommended": pick["method"], "verdict": "recommended", "rank_key": pref,
            "tied": sorted(r["method"] for r in tied)}


def downstream(recs: list, dataset: str) -> dict:
    """Q4: c of the closest own rung, the fellow eye and the mean (P and N), with contrasts."""
    ok = [r for r in recs if not r["fail"]["anchor"] and not r["dropped"]]
    own = [r for r in ok if r["rung"] == CLOSEST[dataset]]
    fel = [r for r in ok if fellow(r)]
    pe = lambda rs: [r["person"] for r in rs]  # noqa: E731
    cc = {}
    out = {}
    for rep in ("P", "N"):
        out[f"c_own_{rep}"] = boot([r["c"][f"own_{rep}"] for r in own], pe(own), cc)
        out[f"c_fellow_{rep}"] = boot([r["c"][f"own_{rep}"] for r in fel], pe(fel), cc)
        out[f"c_mean_{rep}"] = boot([r["c"][f"mean_{rep}"] for r in own], pe(own), cc)
        out[f"c_mean_minus_own_{rep}"] = boot([r["c"][f"mean_{rep}"] - r["c"][f"own_{rep}"] for r in own],
                                              pe(own), cc)
        out[f"c_own_minus_fellow_{rep}"] = boot_unpaired([r["c"][f"own_{rep}"] for r in own], pe(own),
                                                         [r["c"][f"own_{rep}"] for r in fel], pe(fel))
    return out


def versus_none(recs: list, none_recs: list, dataset: str) -> dict:
    """Q4 contrast: method minus G0 / I0 on the same pairs (paired), closest own rung, P."""
    base = {r["pair"]: r for r in none_recs if not r["fail"]["anchor"] and not r["dropped"]}
    rs = [r for r in recs if not r["fail"]["anchor"] and not r["dropped"] and r["pair"] in base
          and r["rung"] == CLOSEST[dataset]]
    cc = {}
    return {f"c_{k}_P_minus_none": boot([r["c"][f"{k}_P"] - base[r["pair"]]["c"][f"{k}_P"] for r in rs],
                                        [r["person"] for r in rs], cc) for k in ("own", "mean")}
