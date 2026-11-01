import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ladder2d_select as ls  # noqa: E402


def grape_table():
    """Toy GRAPE table: person toyA with visits at 0, 0.5 and 2 years (camera change at 2 y); 80 one-visit strangers."""
    rows = []

    def add(pid, side, t, cam):
        rows.append({"Subject Number": pid, "Laterality": side, "Visit Number": 1, "Interval Years": t, "IOP": 15,
                     "Corresponding CFP": f"{pid}_{side}_{t}.jpg", "Acquisition Device": cam, "Resolusion": "x",
                     "Age": 40, "Gender": "F"})

    for t, cam in ((0.0, "A"), (0.5, "A"), (2.0, "B")):
        add("toyA", "OD", t, cam)
        add("toyA", "OS", t, cam)
    for i in range(80):
        for side in ("OD", "OS"):
            add(f"s{i:03d}", side, 0.0, "A" if i % 4 else "B")
    return pd.DataFrame(rows)


def test_grape_pairs_one_per_unit_per_rung_with_anchor_rules():
    u = ls.grape_universe(grape_table())
    prs, _ = ls.pairs(u, ls.grape_rung, ls.GRAPE_RUNGS, np.random.default_rng(0))
    a = prs[prs.person == "toyA"]
    assert set(a.rung) == {"same_camera_<1y", "other_camera", "fellow_eye_same_visit"}
    assert a.groupby(["rung", "unit"]).size().max() == 1
    same = a[a.rung != "fellow_eye_same_visit"]
    assert (same.gap > 0).all()  # anchor is the earlier visit
    fellow = a[a.rung == "fellow_eye_same_visit"]
    assert len(fellow) == 1 and "_OD_" in fellow.anchor.iloc[0] and "_OS_" in fellow.candidate.iloc[0]


def test_strangers_and_mean_members_are_matched_disjoint_and_exclude_the_person():
    u = ls.grape_universe(grape_table())
    rng = np.random.default_rng(1)
    prs, _ = ls.pairs(u, ls.grape_rung, ls.GRAPE_RUNGS, rng)
    recs, dropped = ls.attach(prs, u, ls.CFG["grape"], rng)
    rows = u.set_index("image")
    assert recs
    for r in recs:
        cand = rows.loc[r["candidate"]]
        drawn = rows.loc[r["mean_members"] + r["reference"]]
        assert len(r["mean_members"]) == 16 and len(r["reference"]) >= 30
        assert (drawn.device == cand.device).all() and (drawn.side == cand.side).all()
        assert drawn.person.is_unique and r["person"] not in set(drawn.person)
    # camera B has 20 strangers per side: below M + k_min = 46, so the other-camera pair (candidate on B) is dropped
    assert any("other_camera" in k for k in dropped)


def test_selection_is_deterministic():
    u = ls.grape_universe(grape_table())
    assert ls.select(u, ls.grape_rung, ls.GRAPE_RUNGS, ls.CFG["grape"], 7) == \
        ls.select(u, ls.grape_rung, ls.GRAPE_RUNGS, ls.CFG["grape"], 7)


def test_rung_rules():
    def img(unit, side, device, time):
        return {"unit": unit, "side": side, "device": device, "time": time}

    g = ls.grape_rung
    assert g(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "A", 0.5)) == "same_camera_<1y"
    assert g(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "A", 1.0)) == "same_camera_1y+"
    assert g(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "B", 0.5)) == "other_camera"
    assert g(img("1-OD", "OD", "A", 0.0), img("1-OS", "OS", "A", 0.0)) == "fellow_eye_same_visit"
    assert g(img("1-OD", "OD", "A", 0.0), img("1-OS", "OS", "A", 1.0)) is None
    assert g(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "A", 0.0)) is None
    i = ls.iris_rung
    assert i(img("1-R", "R", "NIR", 0.0), img("1-R", "R", "NIR", 0.0)) == "same_eye_same_sitting"
    assert i(img("1-R", "R", "NIR", 0.0), img("1-L", "L", "NIR", 0.0)) == "fellow_eye_same_sitting"
