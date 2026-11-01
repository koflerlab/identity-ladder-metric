import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import real_cases_select as rs  # noqa: E402


def person(pid, fam, age, ses1="1", ses2="n/a", group="n/a"):
    return {"participant_id": pid, "family_id": fam, "age": str(age), "ses01_T1w_acq": ses1,
            "ses02_T1w_acq": ses2, "group": group}


def toy_rows():
    rows = [person("a1", "A", 20), person("a2", "A", 23), person("a3", "A", 20),  # twins plus an older sibling
            person("b1", "B", 22, group="1", ses2="1"), person("b2", "B", 22),  # retest family, twins only
            person("c1", "C", 21), person("c2", "C", 25, ses1="2"),  # sibling scanned sagittal: not eligible
            person("d1", "D", 16), person("d2", "D", 19)]  # adolescent: not eligible
    rows += [person(f"p{i}", f"P{i}", 24) for i in range(120)]
    return rows


def test_selection_rules():
    sel = rs.select(toy_rows(), seed=1)
    assert [e["family"] for e in sel["set_S"]] == ["A"]
    s = sel["set_S"][0]
    assert {s["target"], s["sibling"]} <= {"a1", "a2", "a3"} and s["age_gap"] == 3
    assert "a2" in (s["target"], s["sibling"])  # the different-age member is always on one side
    assert [e["target"] for e in sel["set_R"]] == ["b1"]
    pools = sel["stranger_pool"] + sel["mean_pool"]
    assert len(sel["stranger_pool"]) == 40 and len(sel["mean_pool"]) == 60 and len(set(pools)) == 100
    assert not set(pools) & {"a1", "a2", "a3", "b1", "b2", "d1", "c2"}  # no selected family, no ineligible person


def test_expanded_selection_rules():
    rows = toy_rows() + [person(f"q{i}{k}", f"Q{i}", 26) for i in range(8) for k in "xy"]
    sel = rs.select_expanded(rows, seed=3, n_strangers=5)
    by_fam = {e["family"]: e for e in sel["targets"]}
    assert sorted(by_fam) == ["A", "B"] + [f"Q{i}" for i in range(8)]  # C: c2 is sagittal; D: d1 is 16
    a = by_fam["A"]
    assert a["target"] in ("a1", "a3") and a["sibling"] == "a2" and a["age_gap"] == 3  # prefers a twin with a sibling
    assert a["cotwin"] == ({"a1", "a3"} - {a["target"]}).pop()
    b = by_fam["B"]
    assert "sibling" not in b and {b["target"], b["cotwin"]} == {"b1", "b2"} and b["repeat"] == (b["target"] == "b1")
    for e in sel["targets"]:
        own = {r["participant_id"] for r in rows if r["family_id"] == e["family"]}
        assert len(e["strangers"]) == 5 and not set(e["strangers"]) & own
        assert not set(e["strangers"]) & set(sel["mean_pool"])
    assert "c1" in sel["mean_pool"] and "d2" in sel["mean_pool"] and len(sel["mean_pool"]) == 122
    assert sorted(sel["processing_order"]) == sorted(by_fam)
    assert rs.select_expanded(rows, seed=3, n_strangers=5) == sel
    need = rs.scans_expanded(rows)
    assert ("b1", "ses-02") in need and ("c2", "ses-01") not in need and ("d1", "ses-01") not in need


def test_selection_is_deterministic_and_lists_every_scan():
    a, b = rs.select(toy_rows()), rs.select(toy_rows())
    assert a == b
    need = rs.scans(a)
    assert ("b1", "ses-02") in need and len(need) == 2 + 2 + 100
