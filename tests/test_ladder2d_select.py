import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ladder2d_select as ls  # noqa: E402


def cxr_meta():
    """Toy Data_Entry table: patient A with gaps 0, 1, 3 and 7 years; B with an AP image and a child image; strangers."""
    rows = []

    def add(pid, follow, age, view="PA", sex="M", spacing=0.143, finding="No Finding"):
        rows.append({"Image Index": f"{pid}_{follow:03d}.png", "Finding Labels": finding, "Follow-up #": follow,
                     "Patient ID": pid, "Patient Age": age, "Patient Sex": sex, "View Position": view,
                     "OriginalImage[Width": 2048, "Height]": 2048, "OriginalImagePixelSpacing[x": spacing, "y]": spacing})

    for f, age in enumerate([40, 40, 41, 43, 47]):
        add("toyA", f, age)
    add("toyB", 0, 30)
    add("toyB", 1, 30, view="AP")
    add("toyB", 2, 12)
    for i in range(300):
        add(f"s{i:04d}", 0, 38 + i % 12, sex="M" if i % 3 else "F", spacing=0.143 if i % 5 else 0.168)
    return pd.DataFrame(rows)


def test_cxr_universe_keeps_adult_pa_images_of_the_downloaded_archives():
    u = ls.cxr_universe(cxr_meta(), n_images=10_000)
    assert set(u.view) == {"PA"} and u.age.min() >= 18
    assert len(u[u.patient == "toyB"]) == 1  # AP and child image removed
    assert len(ls.cxr_universe(cxr_meta(), n_images=5)) == 5  # sorted index order, first n images only


def test_cxr_pairs_rungs_anchor_and_one_pair_per_patient_per_rung():
    u = ls.cxr_universe(cxr_meta(), n_images=10_000)
    pairs, _ = ls.cxr_pairs(u, np.random.default_rng(0))
    a = pairs[pairs.patient == "toyA"].set_index("rung")
    assert sorted(a.index) == ["long_5y+", "mid_2-4y", "short_0y"]
    assert a.loc["short_0y", "gap"] == 0 and 2 <= a.loc["mid_2-4y", "gap"] <= 4 and a.loc["long_5y+", "gap"] >= 5
    order = {img: i for i, img in enumerate(sorted(u.image))}
    assert all(order[x] < order[y] for x, y in zip(pairs.anchor, pairs.candidate))  # anchor = earlier follow-up
    assert not (pairs.gap == 1).any()


def test_stranger_draw_matches_keys_age_excludes_patient_and_widens_once():
    u = ls.cxr_universe(cxr_meta(), n_images=10_000)
    pool = ls.StrangerPool(u, ("sex", "device"))
    row = u[u.image == "toyA_004.png"].iloc[0]  # male, 0.143, age 47
    rng = np.random.default_rng(1)
    imgs, tol = pool.draw(row, {"toyA"}, n=200, n_min=10, age_tols=(2, 5), rng=rng)
    got = u.set_index("image").loc[imgs]
    assert tol == 2 and (got.sex == "M").all() and (got.device == "0.143").all()
    assert (abs(got.age - 47) <= 2).all() and "toyA" not in set(got.patient) and got.patient.is_unique
    matched = u[(u.sex == "M") & (u.device == "0.143") & (u.patient != "toyA")]
    n2, n5 = (int((abs(matched.age - 47) <= t).sum()) for t in (2, 5))
    assert n2 < n5
    imgs, tol = pool.draw(row, {"toyA"}, n=200, n_min=n2 + 1, age_tols=(2, 5), rng=rng)
    assert tol == 5 and len(imgs) == n5
    assert pool.draw(row, {"toyA"}, n=200, n_min=10_000, age_tols=(2, 5), rng=rng) == (None, None)


def test_attach_mean_members_and_reference_are_disjoint_strangers():
    u = ls.cxr_universe(cxr_meta(), n_images=10_000)
    rng = np.random.default_rng(2)
    pairs, _ = ls.cxr_pairs(u, rng)
    cfg = {**ls.CXR, "k": 40, "k_min": 10, "m": 8}
    recs, _ = ls.attach(pairs, u, u, cfg, rng)
    pat = u.set_index("image").patient
    for r in recs:
        assert r["patient"] not in set(pat[r["ref_candidate"]])
        if r["mean_members"] is not None:
            assert len(r["mean_members"]) == 8
            assert not set(pat[r["mean_members"]]) & set(pat[r["ref_mean"]])
            assert r["patient"] not in set(pat[r["mean_members"]]) | set(pat[r["ref_mean"]])


def test_selection_is_deterministic():
    a = ls._select(ls.cxr_universe(cxr_meta(), 10_000), {"p": (ls.cxr_universe(cxr_meta(), 10_000),) * 2
                                                          + ({**ls.CXR, "k_min": 5},)}, ls.cxr_pairs, 7)
    b = ls._select(ls.cxr_universe(cxr_meta(), 10_000), {"p": (ls.cxr_universe(cxr_meta(), 10_000),) * 2
                                                          + ({**ls.CXR, "k_min": 5},)}, ls.cxr_pairs, 7)
    assert a == b and a["variants"]["p"]["pairs"]


def test_grape_rungs():
    def img(unit, side, device, time):
        return {"unit": unit, "side": side, "device": device, "time": time}

    assert ls.grape_rung(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "A", 0.5)) == "same_camera_<1y"
    assert ls.grape_rung(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "A", 1.0)) == "same_camera_1y+"
    assert ls.grape_rung(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "A", 3.5)) == "same_camera_1y+"
    assert ls.grape_rung(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "B", 0.5)) == "other_camera"
    assert ls.grape_rung(img("1-OD", "OD", "A", 0.0), img("1-OS", "OS", "A", 0.0)) == "fellow_eye_same_visit"
    assert ls.grape_rung(img("1-OD", "OD", "A", 0.0), img("1-OS", "OS", "A", 1.0)) is None
    assert ls.grape_rung(img("1-OD", "OD", "A", 0.0), img("1-OD", "OD", "A", 0.0)) is None
