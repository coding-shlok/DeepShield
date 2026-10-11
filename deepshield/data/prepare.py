"""Stream a dataset source -> face crops + manifest.csv, resumably.

    deepshield prepare --config configs/data/celebdf.yaml

Progress is appended to `<out>/manifest.partial.csv` after every sample, so a killed PBS
job resumes where it stopped. Raw videos only exist in a temp dir while being processed.
"""
from __future__ import annotations

import csv
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .extract import process_images, process_video
from .face_detect import build_detector
from .manifest import assign_splits, read_manifest, summarize, write_manifest
from .sources import build_source

FIELDS = ["sample_id", "dataset", "label", "method", "group", "path", "n_frames", "split"]


def prepare(cfg: dict) -> Path:
    name = cfg["dataset"]
    out = Path(cfg.get("out_dir", f"data/processed/{name}"))
    (out / "faces").mkdir(parents=True, exist_ok=True)
    partial = out / "manifest.partial.csv"
    done = set()
    if partial.exists():
        done = set(pd.read_csv(partial, dtype=str)["sample_id"])
        print(f"[prepare] resuming: {len(done)} samples already processed")

    src_cfg = dict(cfg.get("source", {}))
    source = build_source(name, **src_cfg)
    det_name = cfg.get("detector") or getattr(source, "default_detector", "retinaface")
    detector = build_detector(det_name, cfg.get("device", "cuda"))
    n_frames = cfg.get("n_frames", 32)
    margin, size = cfg.get("margin", 1.3), cfg.get("crop_size", 256)
    limit = cfg.get("limit")
    workers = cfg.get("io_workers", 4)

    new_file = not partial.exists()
    fh = open(partial, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    n_ok = n_fail = 0
    t0 = time.time()

    def work(item, data, tmp):
        rel = f"faces/{item.sample_id}"
        try:
            if item.kind == "video":
                n = process_video(data, out / rel, detector, n_frames, margin, size)
            else:
                n = process_images(list(data), out / rel, detector, margin, size,
                                   n_frames=n_frames if len(data) > 1 else None)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # raw video gone as soon as it is processed
        return item, rel, n

    # Detection runs on one GPU context; threads overlap network/decoding with detection.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for item in source.iter_items():
            if item.sample_id in done:
                continue
            if limit and n_ok + len(futures) >= limit:
                break
            # fetch in this thread: archive members must be read while the source still has them open
            tmp = tempfile.mkdtemp(prefix="ds_item_", dir=src_cfg.get("tmp_dir"))
            try:
                data = item.fetch(Path(tmp))
            except Exception as e:
                print(f"[prepare] fetch failed for {item.sample_id}: {e}")
                shutil.rmtree(tmp, ignore_errors=True)
                n_fail += 1
                continue
            futures.append(ex.submit(work, item, data, tmp))
            if len(futures) >= workers * 2:
                n_ok, n_fail = _drain(futures, writer, fh, name, n_ok, n_fail, keep=workers)
        n_ok, n_fail = _drain(futures, writer, fh, name, n_ok, n_fail, keep=0)
    fh.close()
    print(f"[prepare] {name}: +{n_ok} ok, {n_fail} failed in {time.time() - t0:.0f}s")
    return finalize(out, cfg)


def _drain(futures, writer, fh, name, n_ok, n_fail, keep):
    while len(futures) > keep:
        f = futures.pop(0)
        try:
            item, rel, n = f.result()
        except Exception as e:
            print(f"[prepare] failed: {e}")
            n_fail += 1
            continue
        if n == 0:
            n_fail += 1
            continue
        writer.writerow(dict(sample_id=item.sample_id, dataset=name, label=item.label, method=item.method,
                             group=item.group or item.sample_id, path=rel, n_frames=n, split=item.split))
        fh.flush()
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"[prepare] {name}: {n_ok} processed")
    return n_ok, n_fail


def finalize(out: Path, cfg: dict) -> Path:
    partial = out / "manifest.partial.csv"
    df = pd.read_csv(partial, dtype={"sample_id": str, "group": str, "method": str})
    df = df.drop_duplicates("sample_id", keep="last")
    df["split"] = df["split"].fillna("")
    ratios = tuple(cfg.get("split_ratios", (0.70, 0.15, 0.15)))
    # official splits often have only train/test -> carve val out of train groups
    if (df["split"] == "test").any() and not (df["split"] == "val").any():
        df.loc[df["split"] == "train", "split"] = ""
        r = ratios[0] + ratios[1]
        df_nt = assign_splits(df[df["split"] != "test"], (ratios[0] / r, ratios[1] / r, 0.0), cfg.get("seed", 42))
        df_nt.loc[df_nt["split"] == "test", "split"] = "val"
        df.loc[df_nt.index, "split"] = df_nt["split"]
    df = assign_splits(df, ratios, seed=cfg.get("seed", 42))
    write_manifest(df, out / "manifest.csv")
    print(summarize(read_manifest(out / "manifest.csv")))
    return out / "manifest.csv"


def check_processed(manifest: str | Path, verify_images: bool = False) -> dict:
    """Integrity check: missing dirs, empty samples, (optionally) undecodable JPEGs."""
    import cv2

    df = read_manifest(manifest)
    root = Path(df.attrs["root"])
    missing, empty, corrupt = [], [], []
    for _, r in df.iterrows():
        p = root / r["path"]
        if not p.exists():
            missing.append(r["sample_id"]); continue
        files = sorted(p.iterdir()) if p.is_dir() else [p]
        if not files:
            empty.append(r["sample_id"]); continue
        if verify_images:
            for f in files:
                if cv2.imread(str(f)) is None:
                    corrupt.append(str(f))
    rep = dict(samples=len(df), missing=len(missing), empty=len(empty), corrupt_files=len(corrupt),
               per_split={f"{sp}/{'fake' if lb else 'real'}": int(n)
                          for (sp, lb), n in df.groupby(["split", "label"]).size().items()})
    rep["ok"] = not (missing or empty or corrupt)
    rep["examples"] = (missing + empty + corrupt)[:10]
    return rep
