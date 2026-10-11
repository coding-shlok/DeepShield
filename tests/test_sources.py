"""Dataset adapters on fake archives laid out like the real releases."""
import io
import json
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from deepshield.data.sources import build_source


def _video_bytes(tmp: Path, n=8, color=(0, 128, 255)) -> bytes:
    p = tmp / "v.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(n):
        f = np.full((64, 64, 3), color, np.uint8)
        cv2.circle(f, (32, 32), 10 + i, (255, 255, 255), -1)
        vw.write(f)
    vw.release()
    return p.read_bytes()


def _jpg() -> bytes:
    return cv2.imencode(".jpg", np.random.randint(0, 255, (40, 40, 3), np.uint8))[1].tobytes()


def test_dfdc_zip(tmp_path):
    v = _video_bytes(tmp_path)
    zp = tmp_path / "dfdc_train_part_0.zip"
    meta = {"a.mp4": {"label": "REAL", "split": "train"}, "b.mp4": {"label": "FAKE", "original": "a.mp4", "split": "train"}}
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("dfdc_train_part_0/a.mp4", v)
        z.writestr("dfdc_train_part_0/b.mp4", v)
        z.writestr("dfdc_train_part_0/metadata.json", json.dumps(meta))  # metadata last on purpose
    items, sizes = [], []
    for i in build_source("dfdc", locations=[str(zp)]).iter_items():
        items.append(i)
        sizes.append(Path(i.fetch(tmp_path)).stat().st_size)  # must fetch while the archive is open
    assert {(i.sample_id, i.label, i.group) for i in items} == {("a", 0, "a"), ("b", 1, "a")}
    assert all(s > 0 for s in sizes)


def test_celebdf_and_ffpp_dirs(tmp_path):
    v = _video_bytes(tmp_path)
    root = tmp_path / "celeb"
    for rel in ["Celeb-real/id0_0000.mp4", "Celeb-synthesis/id0_id1_0000.mp4", "YouTube-real/00001.mp4"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(v)
    (root / "List_of_testing_videos.txt").write_text("1 YouTube-real/00001.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n")
    items = {i.sample_id: i for i in build_source("celebdf", locations=[str(root)]).iter_items()}
    assert items["Celeb-synthesis_id0_id1_0000"].label == 1 and items["Celeb-synthesis_id0_id1_0000"].split == "test"
    assert items["Celeb-real_id0_0000"].split == ""

    ff = tmp_path / "ff"
    for rel in ["original_sequences/youtube/c23/videos/000.mp4", "manipulated_sequences/Deepfakes/c23/videos/000_003.mp4",
                "manipulated_sequences/Deepfakes/c40/videos/000_003.mp4"]:
        (ff / rel).parent.mkdir(parents=True, exist_ok=True)
        (ff / rel).write_bytes(v)
    src = build_source("faceforensics", locations=[str(ff)])
    items = list(src.iter_items())
    assert sorted((i.label, i.method) for i in items) == [(0, "real"), (1, "Deepfakes")]  # c40 skipped
    assert len({i.group for i in items}) == 1  # original and its fake share a group


def test_wilddeepfake_tar_and_prepare(tmp_path):
    from deepshield.data.prepare import check_processed, prepare

    arch = tmp_path / "fake_train.tar.gz"
    with tarfile.open(arch, "w:gz") as tf:
        for vid in ("v1", "v2"):
            for k in range(5):
                data = _jpg()
                ti = tarfile.TarInfo(f"fake_train/{vid}/seq0/{k}.jpg")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    items = list(build_source("wilddeepfake", locations=[str(arch)]).iter_items())
    assert len(items) == 2 and all(i.label == 1 and i.kind == "images" for i in items)

    # full prepare on a generic folder dataset with videos (resumable manifest)
    v = _video_bytes(tmp_path)
    raw = tmp_path / "raw"
    for k in range(6):
        for lab in ("real", "fake"):
            (raw / lab).mkdir(parents=True, exist_ok=True)
            (raw / lab / f"{k}.mp4").write_bytes(v)
    out = tmp_path / "proc"
    cfg = {"dataset": "uadfv", "out_dir": str(out), "detector": "none", "device": "cpu", "n_frames": 4,
           "crop_size": 32, "io_workers": 2, "source": {"locations": [str(raw)]}}
    manifest = prepare(cfg)
    df = pd.read_csv(manifest)
    assert len(df) == 12 and set(df.label) == {0, 1} and set(df.split) <= {"train", "val", "test"}
    assert (df.n_frames == 4).all()
    assert check_processed(manifest, verify_images=True)["ok"]
    # resume: nothing new processed, manifest identical
    assert len(pd.read_csv(prepare(cfg))) == 12


def _range_server(directory: Path):
    import functools
    import http.server
    import threading

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Accept-Ranges", "bytes")
            super().end_headers()

        def do_GET(self):
            rng = self.headers.get("Range")
            if not rng:
                return super().do_GET()
            data = (directory / self.path.lstrip("/")).read_bytes()
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            self.send_response(206)
            self.send_header("Content-Length", str(b - a + 1))
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            self.end_headers()
            self.wfile.write(data[a:b + 1])

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=str(directory)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_remote_zip_range_and_tar_stream(tmp_path, monkeypatch):
    from deepshield.data.prepare import prepare

    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    v = _video_bytes(tmp_path)
    srvdir = tmp_path / "srv"
    srvdir.mkdir()
    meta = {f"{i}.mp4": {"label": "FAKE" if i % 2 else "REAL"} for i in range(6)}
    with zipfile.ZipFile(srvdir / "part.zip", "w") as z:
        for name in meta:
            z.writestr(f"part/{name}", v)
        z.writestr("part/metadata.json", json.dumps(meta))
    with tarfile.open(srvdir / "real_test.tar.gz", "w:gz") as tf:
        for k in range(3):
            data = _jpg()
            ti = tarfile.TarInfo(f"real_test/v0/s0/{k}.jpg")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    srv = _range_server(srvdir)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        out = tmp_path / "dfdc"
        m = prepare({"dataset": "dfdc", "out_dir": str(out), "detector": "none", "device": "cpu", "n_frames": 3,
                     "crop_size": 32, "io_workers": 2, "source": {"locations": [f"{base}/part.zip"]}})
        assert len(pd.read_csv(m)) == 6
        items = list(build_source("wilddeepfake", locations=[f"{base}/real_test.tar.gz"]).iter_items())
        assert len(items) == 1 and items[0].label == 0 and items[0].split == "test"
        assert len(items[0].fetch(tmp_path)) == 3
    finally:
        srv.shutdown()
