"""Deterministic selection of the real-case targets and pools from QTIM metadata (docs/PROTOCOL_real_cases.md).

Reads only participants.tsv (no images). Writes the selection, which holds participant IDs, to the private project
folder of the data root and prints counts and the file's SHA-256 only. Rules (docs/DECISION_TREES.md, D1 to D6):
  adults (age >= 18) scanned coronal at session 1;
  set S: one target per family that has a different-age (non-twin) relative; the sibling is the relative with the
         smallest age gap, ties broken by the seed;
  set R: test-retest participants coronal in both sessions, one per family, not already selected in set S;
  stranger pool (40) and mean pool (60): one person per family, from families with no selected person, disjoint.
"""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

SEED = 20261101
N_STRANGER_POOL, N_MEAN_POOL = 40, 60
DATA_ROOT = os.environ.get("DATA_ROOT", r"C:\Projects\data")
PARTICIPANTS = os.path.join(DATA_ROOT, "brain_images", "siblings", "participants.tsv")
OUT = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", "real-cases-protocol", "selection.json")


def eligible(r):
    return int(r["age"]) >= 18 and r["ses01_T1w_acq"] == "1"


def select(rows, seed=SEED):
    rng = np.random.default_rng(seed)
    fam = defaultdict(list)
    for r in rows:
        if eligible(r):
            fam[r["family_id"]].append(r)
    used, set_s = set(), []
    for fid in sorted(fam):
        members = fam[fid]
        cands = [r for r in members if any(o["age"] != r["age"] for o in members)]
        if not cands:
            continue
        t = cands[rng.integers(len(cands))]
        rel = [o for o in members if o["age"] != t["age"]]
        gap = min(abs(int(o["age"]) - int(t["age"])) for o in rel)
        closest = [o for o in rel if abs(int(o["age"]) - int(t["age"])) == gap]
        s = closest[rng.integers(len(closest))]
        set_s.append({"family": fid, "target": t["participant_id"], "sibling": s["participant_id"], "age_gap": gap})
        used |= {t["participant_id"], s["participant_id"]}
    set_r, r_fams = [], set()
    for fid in sorted(fam):
        cands = [r for r in fam[fid] if r["group"] == "1" and r["ses02_T1w_acq"] == "1" and r["participant_id"] not in used]
        if cands:
            t = cands[rng.integers(len(cands))]
            set_r.append({"family": fid, "target": t["participant_id"]})
            used.add(t["participant_id"])
            r_fams.add(fid)
    busy = {e["family"] for e in set_s} | r_fams
    free = [fid for fid in sorted(fam) if fid not in busy]
    order = rng.permutation(len(free))
    picks = [fam[free[i]][rng.integers(len(fam[free[i]]))]["participant_id"] for i in order[:N_STRANGER_POOL + N_MEAN_POOL]]
    return {"seed": seed, "set_S": set_s, "set_R": set_r, "stranger_pool": picks[:N_STRANGER_POOL],
            "mean_pool": picks[N_STRANGER_POOL:]}


def scans(sel):
    """Every (participant, session) T1w scan the analysis needs."""
    need = {(e["target"], "ses-01") for e in sel["set_S"]} | {(e["sibling"], "ses-01") for e in sel["set_S"]}
    need |= {(e["target"], s) for e in sel["set_R"] for s in ("ses-01", "ses-02")}
    need |= {(p, "ses-01") for p in sel["stranger_pool"] + sel["mean_pool"]}
    return sorted(need)


def select_expanded(rows, seed=SEED, n_strangers=16):
    """Amendment 1, A1.1: one target per family with 2+ eligible members, co-twin (same age) and sibling (different
    age, closest) rows, a fixed mean pool of all single-member families, 16 strangers per target from other
    multi-member families, and a seeded processing order of the target families."""
    rng = np.random.default_rng(seed)
    fam = defaultdict(list)
    for r in rows:
        if eligible(r):
            fam[r["family_id"]].append(r)
    multi = sorted(f for f in fam if len(fam[f]) >= 2)
    single = sorted(f for f in fam if len(fam[f]) == 1)
    targets = []
    for fid in multi:
        members = fam[fid]
        kinds = lambda r: ({o["age"] == r["age"] for o in members if o is not r})  # True: same age, False: other
        both = [r for r in members if kinds(r) == {True, False}]
        pool = both or members
        t = pool[rng.integers(len(pool))]
        twins = [o for o in members if o is not t and o["age"] == t["age"]]
        sibs = [o for o in members if o["age"] != t["age"]]
        e = {"family": fid, "target": t["participant_id"],
             "repeat": t["group"] == "1" and t["ses02_T1w_acq"] == "1"}
        if twins:
            e["cotwin"] = twins[rng.integers(len(twins))]["participant_id"]
        if sibs:
            gap = min(abs(int(o["age"]) - int(t["age"])) for o in sibs)
            closest = [o for o in sibs if abs(int(o["age"]) - int(t["age"])) == gap]
            e["sibling"], e["age_gap"] = closest[rng.integers(len(closest))]["participant_id"], gap
        others = [f for f in multi if f != fid]
        e["strangers"] = [fam[f][rng.integers(len(fam[f]))]["participant_id"]
                          for f in (others[i] for i in rng.choice(len(others), n_strangers, replace=False))]
        targets.append(e)
    order = [targets[i]["family"] for i in rng.permutation(len(targets))]
    return {"seed": seed, "targets": targets, "processing_order": order,
            "mean_pool": [fam[f][0]["participant_id"] for f in single]}


def scans_expanded(rows):
    """A1.1 downloads every eligible session-1 scan and the session-2 scan of every coronal test-retest participant."""
    el = [r for r in rows if eligible(r)]
    return sorted({(r["participant_id"], "ses-01") for r in el} |
                  {(r["participant_id"], "ses-02") for r in el if r["group"] == "1" and r["ses02_T1w_acq"] == "1"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=PARTICIPANTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--expanded", action="store_true", help="Amendment 1 (A1.1) selection")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.participants, encoding="utf-8"), delimiter="\t"))
    if a.expanded:
        out = a.out if a.out != OUT else OUT.replace("selection.json", "selection_expanded.json")
        sel = select_expanded(rows)
        sel["scans"] = [list(x) for x in scans_expanded(rows)]
        blob = json.dumps(sel, indent=1, sort_keys=True).encode("utf-8")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(blob)
        t = sel["targets"]
        print({"targets": len(t), "with_cotwin": sum("cotwin" in e for e in t), "with_sibling": sum("sibling" in e for e in t),
               "with_both": sum("cotwin" in e and "sibling" in e for e in t), "with_repeat": sum(e["repeat"] for e in t),
               "mean_pool": len(sel["mean_pool"]), "scans": len(sel["scans"]),
               "selection_expanded_sha256": hashlib.sha256(blob).hexdigest()})
        return
    sel = select(rows)
    sel["scans"] = [list(x) for x in scans(sel)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    blob = json.dumps(sel, indent=1, sort_keys=True).encode("utf-8")
    with open(a.out, "wb") as f:
        f.write(blob)
    gaps = [e["age_gap"] for e in sel["set_S"]]
    print({"set_S": len(sel["set_S"]), "set_R": len(sel["set_R"]), "stranger_pool": len(sel["stranger_pool"]),
           "mean_pool": len(sel["mean_pool"]), "scans": len(sel["scans"]), "sibling_age_gap_min_max": [min(gaps), max(gaps)],
           "participants_sha256": hashlib.sha256(open(a.participants, "rb").read()).hexdigest(),
           "selection_sha256": hashlib.sha256(blob).hexdigest()})


if __name__ == "__main__":
    main()
