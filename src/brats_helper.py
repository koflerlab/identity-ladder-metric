"""Run inside external/venv-brats (protocol A1.7: kept apart from the frozen environment): BraTS-style
preprocessing and challenge-model inference for claim C10 (D11).

  preprocess T1 OUT       rigid registration to the BraTS SRI24 atlas with atlas correction and HD-BET brain
                          extraction on CPU (brainles-preprocessing); writes OUT/t1n_bet.nii.gz, OUT/bet_mask.nii.gz
                          and OUT/transforms/
  infer ALGO T1N MASK OUT one challenge model (brats Inpainter, force_cpu=True) on a voided T1n and its mask

Windows shim (no vendor file is changed): brats imports spython, whose terminal helpers import the Unix-only `pwd`
module for Singularity, which is not used here; its Docker backend calls os.getuid()/os.getgid() for the container
user. A stub `pwd` module and uid/gid 0 (root in the container, as Docker Desktop maps bind mounts anyway) let the
Docker backend run unchanged. Inputs are keyed files; nothing here prints participant IDs.
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path


def windows_shim():
    if os.name == "nt":
        sys.modules.setdefault("pwd", types.ModuleType("pwd"))
        os.getuid = getattr(os, "getuid", lambda: 0)
        os.getgid = getattr(os, "getgid", lambda: 0)


def preprocess(t1, out):
    from brainles_preprocessing.brain_extraction import HDBetExtractor
    from brainles_preprocessing.modality import CenterModality
    from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
    from brainles_preprocessing.registration import ANTsRegistrator
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    center = CenterModality("t1n", t1, raw_bet_output_path=out / "t1n_bet.nii.gz",
                            bet_mask_output_path=out / "bet_mask.nii.gz", atlas_correction=True)
    pre = AtlasCentricPreprocessor(center_modality=center, moving_modalities=[], registrator=ANTsRegistrator(),
                                   brain_extractor=HDBetExtractor(), use_gpu=False)
    start = time.monotonic()
    pre.run(save_dir_transformations=out / "transforms", log_file=out / "preprocess.log")
    return {"seconds": time.monotonic() - start}


def inpainter(algo):
    windows_shim()
    from brats import Inpainter
    from brats.constants import InpaintingAlgorithms
    return Inpainter(algorithm=InpaintingAlgorithms[algo], force_cpu=True)


def infer(algo, t1n, mask, out):
    start = time.monotonic()
    inpainter(algo).infer_single(t1n=t1n, mask=mask, output_file=out)
    return {"seconds": time.monotonic() - start}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preprocess")
    p.add_argument("t1")
    p.add_argument("out")
    i = sub.add_parser("infer")
    for name in ("algo", "t1n", "mask", "out"):
        i.add_argument(name)
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "preprocess":
        res = preprocess(a.t1, a.out)
    elif a.cmd == "infer":
        res = infer(a.algo, a.t1n, a.mask, a.out)
    else:  # compatibility check: the shim imports brats and both C10 models construct on CPU
        res = {algo: type(inpainter(algo)).__name__ for algo in ("BraTS23_1", "BraTS24_1")}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
