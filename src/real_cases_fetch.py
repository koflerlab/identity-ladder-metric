"""Download the T1w scans of the real-case selection from OpenNeuro ds004169 v1.0.8 (docs/PROTOCOL_real_cases.md).

Checks the private selection against its SHA-256 in the protocol and that OpenNeuro serves snapshot 1.0.8, then
fetches each scan through the version-pinned snapshot URL into the canonical `brain_images/siblings` tree: a `.part`
file, verified by size and by the MD5 of the single-part S3 ETag, then renamed. An existing scan is never
overwritten. Writes the private scan manifest (SHA-256 per file) next to the selection. Runs under the data-root
maintenance lock (`C:\\Projects\\data\\AGENTS.md`). Prints counts only, never participant IDs.
"""
import argparse
import datetime
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA_ROOT = os.environ.get("DATA_ROOT", r"C:\Projects\data")
BIDS = os.path.join(DATA_ROOT, "brain_images", "siblings")
PRIVATE = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", "real-cases-protocol")
LOCK = os.path.join(DATA_ROOT, "_registry", "storage-maintenance.lock")
DESIGNS = {  # selection file, its SHA-256 in the protocol, manifest written after the download
    "frozen": ("selection.json", "cb264436a1346f6a83b17e7d3abd8f3bc27ac10174ad89c8d7339eec421f7f31",
               "scan_manifest.json"),
    "expanded": ("selection_expanded.json", "fcfca9875f9b895b1d96ac11b357248ab950a972eed7b7b125e878e83b8d3dc6",
                 "scan_manifest_expanded.json"),  # Amendment 1 (A1.1, A1.5)
}
DATASET, VERSION, DOI = "ds004169", "1.0.8", "doi:10.18112/openneuro.ds004169.v1.0.8"
SNAPSHOT_URL = f"https://openneuro.org/crn/datasets/{DATASET}/snapshots/{VERSION}/files/"
GRAPHQL = "https://openneuro.org/crn/graphql"
LOCK_OWNER = "identity-ladder-metric/src/real_cases_fetch.py"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def relpath(pid, ses, suffix="_T1w.nii.gz"):
    return f"{pid}/{ses}/anat/{pid}_{ses}{suffix}"


def snapshot_served(opener=urllib.request.urlopen):
    """True if OpenNeuro lists snapshot VERSION of DATASET."""
    q = json.dumps({"query": '{ snapshot(datasetId: "%s", tag: "%s") { tag } }' % (DATASET, VERSION)}).encode()
    req = urllib.request.Request(GRAPHQL, data=q, headers={"Content-Type": "application/json"})
    with opener(req, timeout=60) as r:
        snap = (json.load(r).get("data") or {}).get("snapshot") or {}
    return snap.get("tag") == VERSION


def fetch_one(rel, root=BIDS, opener=urllib.request.urlopen):
    """Download one file to `.part`, verify size and ETag MD5, rename; an existing file is hashed, never replaced."""
    dest = os.path.join(root, *rel.split("/"))
    if os.path.exists(dest):
        return {"path": rel, "status": "present", "bytes": os.path.getsize(dest), "sha256": sha256(dest)}
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    md5, sh = hashlib.md5(), hashlib.sha256()
    with opener(SNAPSHOT_URL + rel.replace("/", ":"), timeout=120) as r:
        size = int(r.headers["Content-Length"])
        etag = (r.headers.get("ETag") or "").strip('"')
        with open(part, "wb") as f:
            for block in iter(lambda: r.read(1 << 20), b""):
                f.write(block)
                md5.update(block)
                sh.update(block)
    if os.path.getsize(part) != size:
        raise IOError("size differs from Content-Length")
    md5_checked = bool(etag) and "-" not in etag  # a multipart ETag is not an MD5
    if md5_checked and md5.hexdigest() != etag:
        raise IOError("MD5 differs from ETag")
    os.rename(part, dest)  # raises if dest appeared meanwhile: never overwrite
    return {"path": rel, "status": "downloaded", "bytes": size, "sha256": sh.hexdigest(), "md5_verified": md5_checked}


def acquire_lock(purpose, reuse=False):
    """Exclusive create of the data-root maintenance lock; fails if any agent holds it. With reuse, continue under
    a lock this script already holds (and fail if another owner holds it)."""
    if reuse:
        with open(LOCK, encoding="utf-8") as f:
            if json.load(f).get("owner") != LOCK_OWNER:
                raise SystemExit("lock belongs to another owner")
        return
    body = {"owner": LOCK_OWNER, "pid": os.getpid(), "purpose": purpose,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(LOCK, "x", encoding="utf-8") as f:
        json.dump(body, f)


def release_lock():
    """Remove the lock only if this script created it."""
    with open(LOCK, encoding="utf-8") as f:
        owner = json.load(f).get("owner")
    if owner != LOCK_OWNER:
        raise SystemExit("lock belongs to another owner; not released")
    os.remove(LOCK)


def fetch_all(scans, workers, root=BIDS, opener=urllib.request.urlopen, suffix="_T1w.nii.gz"):
    """Fetch every scan, retrying a failure once; returns (manifest rows, failure count)."""
    def task(rel):
        for attempt in (1, 2):
            try:
                return fetch_one(rel, root, opener)
            except Exception as e:  # noqa: BLE001 - counted and reported, never silent
                err = type(e).__name__
        return {"path": rel, "status": "failed", "error": err}
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(task, [relpath(p, s, suffix) for p, s in scans]))
    return rows, sum(r["status"] == "failed" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--hold-lock", action="store_true", help="keep the lock after success (registry update follows)")
    ap.add_argument("--release-lock", action="store_true", help="only release this script's lock")
    ap.add_argument("--design", choices=list(DESIGNS), default="frozen")
    ap.add_argument("--sidecars", action="store_true", help="fetch the T1w JSON sidecars (data QC) instead of images")
    ap.add_argument("--reuse-lock", action="store_true", help="continue under the lock this script already holds")
    a = ap.parse_args()
    if a.release_lock:
        release_lock()
        print({"lock": "released"})
        return
    name, want, manifest_name = DESIGNS[a.design]
    suffix = "_T1w.json" if a.sidecars else "_T1w.nii.gz"
    if a.sidecars:
        manifest_name = manifest_name.replace("scan_manifest", "sidecar_manifest")
    selection = os.path.join(PRIVATE, name)
    if sha256(selection) != want:
        raise SystemExit(f"{name} differs from the hash in the protocol")
    if not snapshot_served():
        raise SystemExit(f"OpenNeuro does not serve {DATASET} {VERSION}")
    acquire_lock(f"download {DATASET} {VERSION} T1w{' sidecars' if a.sidecars else ' scans'} for "
                 f"identity-ladder-metric real cases ({a.design})", reuse=a.reuse_lock)
    ok = False
    try:
        scans = json.load(open(selection, encoding="utf-8"))["scans"]
        rows, failed = fetch_all(scans, a.workers, suffix=suffix)
        counts = {k: sum(r["status"] == k for r in rows) for k in ("downloaded", "present", "failed")}
        if failed:
            print({"scans": len(rows), **counts, "manifest": "not written"})
            raise SystemExit(1)
        manifest = {"dataset": DATASET, "version": VERSION, "doi": DOI, "source": SNAPSHOT_URL,
                    "retrieved": datetime.date.today().isoformat(), "selection_sha256": want,
                    "files": sorted(rows, key=lambda r: r["path"])}
        blob = json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8")
        with open(os.path.join(PRIVATE, manifest_name), "xb") as f:  # a manifest is written once
            f.write(blob)
        print({"scans": len(rows), **counts, "bytes": sum(r["bytes"] for r in rows),
               "md5_verified": sum(bool(r.get("md5_verified")) for r in rows),
               "manifest_sha256": hashlib.sha256(blob).hexdigest()})
        ok = True
    finally:
        if not (ok and a.hold_lock):
            release_lock()


if __name__ == "__main__":
    main()
