"""Two follow-up checks on GRAPE (aggregate output only):
1. Mirror rung: compare each image with its fellow eye after a horizontal flip (16x16 pixel space), as the left and
   right fundus are mirror images; the released RETFound features cannot be recomputed for flipped images.
2. Camera-time confound: inter-visit interval of same-eye pairs taken with the same vs a different camera.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import reid  # noqa: E402
from run_grape import EXT, IMG_DIR, PROJECT  # noqa: E402


def flipped_pixel_features(names):
    import torchvision.transforms.v2 as T
    from torchvision.io import read_image

    resize = T.Resize((16, 16))
    return {n: (resize(read_image(os.path.join(IMG_DIR, n)).flip(-1)).flatten() / 256.0).numpy().astype(np.float64) for n in names}


def main():
    t = reid.load_table(os.path.join(EXT, "data", "GRAPE", "grape_img_info_w_file.csv"))
    names = t["filename"].tolist()
    plain = reid.pixel_features(IMG_DIR, names)
    flipped = flipped_pixel_features(names)
    info = t.set_index("filename")
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = info.loc[a], info.loc[b]
            rung = reid.pair_rung(pa, pb)
            if rung == "same_eye_same_visit":
                continue
            d_plain = float(np.linalg.norm(plain[a] - plain[b]))
            # mirror one image of every opposite-eye pair (fellow eyes and opposite-eye unrelated pairs alike)
            opposite = pa["eye"] != pb["eye"]
            d_mirror = float(np.linalg.norm(plain[a] - flipped[b])) if opposite else d_plain
            rows.append((rung, bool(opposite), pa["device"] == pb["device"], d_plain, d_mirror,
                         abs(float(pa["interval_years"]) - float(pb["interval_years"])) if pa["patient"] == pb["patient"] else np.nan))
    p = pd.DataFrame(rows, columns=["rung", "opposite_eye", "same_camera", "d_plain", "d_mirror", "gap_years"])

    def sel(df, col):
        return df[col].to_numpy()

    out = {}
    for cam in (True, False):
        s = p[p["same_camera"] == cam]
        fellow = s[s["rung"] == "fellow_eye_same_visit"] if cam else s[s["rung"] == "fellow_eye_other_visit"]
        unrel_opp = s[(s["rung"] == "other_patient") & s["opposite_eye"]]
        same_eye = s[s["rung"] == "same_eye_other_visit"]
        key = "same_camera" if cam else "different_camera"
        out[key] = {
            "fellow_rung_used": "fellow_eye_same_visit" if cam else "fellow_eye_other_visit",
            "n_fellow_pairs": int(len(fellow)),
            "median_fellow_plain": float(np.median(sel(fellow, "d_plain"))) if len(fellow) else None,
            "median_fellow_mirrored": float(np.median(sel(fellow, "d_mirror"))) if len(fellow) else None,
            "median_unrelated_opposite_eye_mirrored": float(np.median(sel(unrel_opp, "d_mirror"))),
            "median_same_eye_other_visit": float(np.median(sel(same_eye, "d_plain"))) if len(same_eye) else None,
            "dprime_fellow_vs_unrelated_plain": reid.decidability(sel(fellow, "d_plain"), sel(unrel_opp, "d_plain")) if len(fellow) > 2 else None,
            "dprime_fellow_vs_unrelated_mirrored": reid.decidability(sel(fellow, "d_mirror"), sel(unrel_opp, "d_mirror")) if len(fellow) > 2 else None,
        }
    se = p[p["rung"] == "same_eye_other_visit"]
    out["camera_time_confound"] = {
        "same_eye_pairs_same_camera_median_gap_years": float(se[se["same_camera"]]["gap_years"].median()),
        "same_eye_pairs_different_camera_median_gap_years": float(se[~se["same_camera"]]["gap_years"].median()),
        "n_same_camera": int(se["same_camera"].sum()), "n_different_camera": int((~se["same_camera"]).sum()),
    }
    res_path = os.path.join(PROJECT, "results", "grape_mirror_interval.json")
    json.dump(out, open(res_path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
