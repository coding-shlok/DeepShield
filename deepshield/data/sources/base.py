"""Location abstraction for streaming datasets with minimal disk usage.

A *location* is any of:
  * a local directory                         /data/raw/Celeb-DF-v2
  * a local archive (.zip/.tar/.tar.gz)       /data/raw/real_train.tar.gz
  * an http(s) URL to an archive              https://.../dfdc_train_part_0.zip
  * a Google-Drive file                       gdrive://<file_id>
  * a Kaggle competition file                 kaggle://deepfake-detection-challenge/train_sample_videos.zip

Remote zips are read with HTTP Range requests when the server supports them, so only the
bytes of the member currently being processed are transferred – nothing is stored. Remote
tars are streamed sequentially. Otherwise the archive is downloaded to a temp dir and deleted
as soon as it has been processed.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Item:
    sample_id: str
    label: int
    kind: str                      # "video" | "images"
    fetch: Callable[[Path], object]  # tmpdir -> local video path | list of image paths/bytes
    method: str = ""
    group: str = ""
    split: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Member:
    """A file inside a location (directory entry or archive member)."""

    name: str                       # posix relative path
    read_bytes: Callable[[], bytes]
    local_path: Path | None = None  # set when it already exists on disk

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.name).suffix.lower()

    def materialize(self, tmpdir: Path) -> Path:
        if self.local_path is not None:
            return self.local_path
        p = Path(tmpdir) / PurePosixPath(self.name).name
        p.write_bytes(self.read_bytes())
        return p


# ----------------------------------------------------------------------------- HTTP range file
class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (lets zipfile read remote archives lazily)."""

    def __init__(self, url: str, block_size: int = 4 << 20, headers: dict | None = None):
        import requests

        self.url, self.block_size = url, block_size
        self.session = requests.Session()
        self.headers = headers or {}
        r = self.session.head(url, allow_redirects=True, headers=self.headers, timeout=60)
        r.raise_for_status()
        self.url = r.url
        if r.headers.get("Accept-Ranges", "").lower() != "bytes" or "Content-Length" not in r.headers:
            raise OSError("server does not support range requests")
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self._cache: dict[int, bytes] = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            start = i * self.block_size
            end = min(self.size, start + self.block_size) - 1
            h = dict(self.headers, Range=f"bytes={start}-{end}")
            for attempt in range(5):
                try:
                    r = self.session.get(self.url, headers=h, timeout=120)
                    r.raise_for_status()
                    break
                except Exception:
                    if attempt == 4:
                        raise
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = r.content
        return self._cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            b = self.pos // self.block_size
            off = self.pos % self.block_size
            chunk = self._block(b)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


# ----------------------------------------------------------------------------- downloads
def download(url: str, dest: Path) -> Path:
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def fetch_remote(loc: str, tmpdir: Path) -> Path:
    if loc.startswith("gdrive://"):
        import gdown

        out = tmpdir / "gdrive_download"
        return Path(gdown.download(id=loc[len("gdrive://"):], output=str(out), quiet=False))
    if loc.startswith("kaggle://"):
        comp, fname = loc[len("kaggle://"):].split("/", 1)
        subprocess.run(["kaggle", "competitions", "download", "-c", comp, "-f", fname, "-p", str(tmpdir)], check=True)
        p = tmpdir / fname
        if not p.exists() and (tmpdir / (fname + ".zip")).exists():
            p = tmpdir / (fname + ".zip")
        return p
    return download(loc, tmpdir / PurePosixPath(loc.split("?")[0]).name)


def is_remote(loc: str) -> bool:
    return loc.startswith(("http://", "https://", "gdrive://", "kaggle://"))


def _is_tar(name: str) -> bool:
    return name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"))


# ----------------------------------------------------------------------------- iteration
def _iter_zip(zf: zipfile.ZipFile) -> Iterator[Member]:
    for info in zf.infolist():
        if info.is_dir():
            continue
        yield Member(info.filename, (lambda i=info: zf.read(i)))


def _iter_tar_stream(fileobj=None, path=None) -> Iterator[Member]:
    mode = "r|*"
    with tarfile.open(name=path, fileobj=fileobj, mode=mode) as tf:
        for m in tf:
            if not m.isfile():
                continue
            data = tf.extractfile(m).read()  # must be consumed now in stream mode
            yield Member(m.name, (lambda d=data: d))


@contextmanager
def open_location(loc: str, tmp_root: str | None = None, keep_downloads: bool = False) -> Iterator[Iterator[Member]]:
    """Context manager yielding an iterator of Members for a location."""
    tmp = Path(tempfile.mkdtemp(prefix="deepshield_", dir=tmp_root))
    try:
        if not is_remote(loc):
            p = Path(loc).expanduser()
            if p.is_dir():
                def it():
                    for f in sorted(p.rglob("*")):
                        if f.is_file():
                            yield Member(f.relative_to(p).as_posix(), f.read_bytes, local_path=f)
                yield it()
            elif zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    yield _iter_zip(zf)
            elif _is_tar(p.name):
                yield _iter_tar_stream(path=str(p))
            else:
                yield iter([Member(p.name, p.read_bytes, local_path=p)])
            return

        name = loc.split("?")[0]
        if loc.startswith(("http://", "https://")) and name.endswith(".zip"):
            try:
                rf = io.BufferedReader(HTTPRangeFile(loc), buffer_size=1 << 20)
                with zipfile.ZipFile(rf) as zf:
                    print(f"[stream] range-reading {loc} (no local copy)")
                    yield _iter_zip(zf)
                return
            except OSError as e:
                print(f"[stream] range requests unavailable ({e}); downloading archive")
        if loc.startswith(("http://", "https://")) and _is_tar(name):
            import requests

            with requests.get(loc, stream=True, timeout=120) as r:
                r.raise_for_status()
                r.raw.decode_content = True
                print(f"[stream] streaming tar {loc} (no local copy)")
                yield _iter_tar_stream(fileobj=r.raw)
            return
        local = fetch_remote(loc, tmp)
        try:
            if zipfile.is_zipfile(local):
                with zipfile.ZipFile(local) as zf:
                    yield _iter_zip(zf)
            elif _is_tar(local.name) or tarfile.is_tarfile(local):
                yield _iter_tar_stream(path=str(local))
            else:
                yield iter([Member(local.name, local.read_bytes, local_path=local)])
        finally:
            if not keep_downloads and local.exists():
                local.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def video_item_from_member(m: Member, **kw) -> Item:
    return Item(kind="video", fetch=lambda tmp, m=m: m.materialize(tmp), **kw)


def env_or(value: str | None, env: str) -> str | None:
    return value or os.environ.get(env)
