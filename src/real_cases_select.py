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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default=PARTICIPANTS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.participants, encoding="utf-8"), delimiter="\t"))
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
