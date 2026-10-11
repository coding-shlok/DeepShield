"""Dataset adapters (PRD §5). Each yields `Item`s with labels, groups and official splits.

All of these datasets are gated behind a licence/request form; you paste the link or
credentials you receive into `configs/data/<dataset>.yaml` (or env vars), and the adapter
streams + processes them. Run `deepshield datasets info` for per-dataset instructions.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterator

from .base import IMAGE_EXT, VIDEO_EXT, Item, Member, open_location, video_item_from_member

DATASET_INFO = {
    "faceforensics": dict(
        stage=2, access="Fill the form at https://github.com/ondyari/FaceForensics#access ; you receive "
        "the download script `faceforensics_download_v4.py`. Point `download_script` at it (c23 videos ~10GB total) "
        "or set `locations` to an already-downloaded root."),
    "dfdc": dict(
        stage=2, access="Accept the licence on Kaggle (deepfake-detection-challenge) or the Meta AI DFDC page. "
        "List the part zips (URLs, kaggle://deepfake-detection-challenge/<file>, or local paths) in `locations`. "
        "Remote zips are range-read so each 10GB part is never stored."),
    "celebdf": dict(
        stage=2, access="Request Celeb-DF v2 via https://github.com/yuezunli/celeb-deepfakeforensics ; put the "
        "Google-Drive id as gdrive://<id> or a local path in `locations`."),
    "deeperforensics": dict(
        stage=3, access="Request via https://github.com/EndlessSora/DeeperForensics-1.0 ; add archive URLs/paths "
        "to `locations`. Real videos: reuse FF++ originals or source_videos."),
    "wilddeepfake": dict(
        stage=3, access="Request via https://github.com/OpenTAI/wild-deepfake ; add the real_*/fake_* tar.gz "
        "archives to `locations` (tar archives are streamed)."),
    "uadfv": dict(stage=4, access="Small dataset (real/ fake/ folders). Set `locations` to its root or archive."),
    "vggface2": dict(
        stage=1, access="VGGFace2 is distributed by the authors on request; any mirror with "
        "<identity>/<image>.jpg layout works. Use `max_per_identity` to bound size."),
    "dffd": dict(stage=1, access="Request DFFD from http://cvlab.cse.msu.edu/dffd-dataset.html ; folder rules "
                 "map ffhq/celeba->real and stylegan/pggan/faceapp/stargan->gan_synthesis."),
    "deepfakebench": dict(
        stage=4, access="Download DeepfakeBench preprocessed data + dataset_json from "
        "https://github.com/SCLBD/DeepfakeBench ; set `json_dir`, `data_root` and the dataset names."),
    "deephy": dict(stage=4, access="Request DeePhy from its authors; configure `rules` for folder labelling."),
    "megammdf": dict(stage=4, access="Mega-MMDF (audio+video); video stream only is used. Configure `rules`."),
    "folder": dict(stage=0, access="Generic: any folder/archive with regex labelling `rules`."),
}


class Source:
    name = "base"
    default_detector = "retinaface"

    def __init__(self, locations=None, tmp_dir=None, keep_downloads=False, limit=None, **kw):
        if isinstance(locations, str):
            locations = [locations]
        self.locations = list(locations or [])
        self.tmp_dir = tmp_dir
        self.keep_downloads = keep_downloads
        self.limit = limit
        self.kw = kw

    def members(self) -> Iterator[tuple[str, Member]]:
        for loc in self.locations:
            with open_location(loc, self.tmp_dir, self.keep_downloads) as it:
                for m in it:
                    yield loc, m

    def iter_items(self) -> Iterator[Item]:
        raise NotImplementedError


# ----------------------------------------------------------------------------- generic folder rules
class FolderSource(Source):
    """Label members by ordered regex rules: [{pattern, label, method?, split?}].

    `group_pattern` (regex with one capture group on the member path) defines identity groups.
    Image members in the same directory become one image-sequence sample when
    `images_as_sequences` is true; otherwise each image is a sample.
    """

    name = "folder"

    def __init__(self, rules=None, group_pattern=None, images_as_sequences=False, dataset_name=None,
                 max_per_group=None, **kw):
        super().__init__(**kw)
        self.rules = [dict(r, _re=re.compile(r["pattern"], re.I)) for r in (rules or [])]
        self.group_re = re.compile(group_pattern) if group_pattern else None
        self.images_as_sequences = images_as_sequences
        self.max_per_group = max_per_group
        if dataset_name:
            self.name = dataset_name

    def _match(self, path: str):
        for r in self.rules:
            if r["_re"].search(path):
                return r
        return None

    def _group(self, path: str, default: str) -> str:
        if self.group_re:
            m = self.group_re.search(path)
            if m:
                return m.group(1)
        return default

    def iter_items(self):
        per_group: dict[str, int] = {}
        seq_key, seq_members, seq_rule = None, [], None

        def flush():
            if seq_members:
                yield self._image_item(seq_key, list(seq_members), seq_rule)

        for _, m in self.members():
            rule = self._match(m.name)
            if rule is None:
                continue
            stem = PurePosixPath(m.name)
            group = self._group(m.name, stem.parent.as_posix() if self.images_as_sequences else stem.stem)
            if self.max_per_group:
                if per_group.get(group, 0) >= self.max_per_group and m.suffix in IMAGE_EXT and not self.images_as_sequences:
                    continue
            if m.suffix in VIDEO_EXT:
                yield from flush(); seq_members.clear()
                per_group[group] = per_group.get(group, 0) + 1
                yield video_item_from_member(m, sample_id=_sid(m.name), label=int(rule["label"]),
                                             method=rule.get("method", ""), group=group, split=rule.get("split", ""))
            elif m.suffix in IMAGE_EXT:
                if self.images_as_sequences:
                    key = stem.parent.as_posix()
                    if key != seq_key:
                        yield from flush(); seq_members.clear()
                        seq_key, seq_rule = key, rule
                    seq_members.append(m)
                else:
                    per_group[group] = per_group.get(group, 0) + 1
                    yield self._image_item(m.name, [m], rule, group)
        yield from flush()

    def _image_item(self, key, members, rule, group=None):
        datas = [mm.read_bytes() for mm in members] if members and members[0].local_path is None else None
        paths = [mm.local_path for mm in members] if datas is None else None
        return Item(sample_id=_sid(key), label=int(rule["label"]), kind="images",
                    fetch=lambda tmp, d=datas, p=paths: d if d is not None else p,
                    method=rule.get("method", ""), group=group or self._group(key, key), split=rule.get("split", ""))


def _sid(path: str) -> str:
    p = PurePosixPath(path)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", p.with_suffix("").as_posix())


# ----------------------------------------------------------------------------- FaceForensics++
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
FFPP_SPLITS = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{}.json"


class FaceForensicsSource(Source):
    name = "faceforensics"

    def __init__(self, compression="c23", methods=None, download_script=None, server="EU2", **kw):
        super().__init__(**kw)
        self.compression = compression
        self.methods = methods or FFPP_METHODS
        self.download_script = download_script
        self.server = server
        self.split_of, self.pair_of = self._load_splits()

    def _load_splits(self):
        split_of, pair_of = {}, {}
        try:
            import requests

            for s in ("train", "val", "test"):
                pairs = requests.get(FFPP_SPLITS.format(s), timeout=30).json()
                for a, b in pairs:
                    key = "_".join(sorted((a, b)))
                    for v in (a, b, f"{a}_{b}", f"{b}_{a}"):
                        split_of[v], pair_of[v] = s, key
        except Exception as e:
            print(f"[ff++] could not fetch official splits ({e}); random group split will be used")
        return split_of, pair_of

    def _item(self, m: Member, method: str) -> Item | None:
        if m.suffix not in VIDEO_EXT or f"/{self.compression}/" not in f"/{m.name}":
            return None
        vid = PurePosixPath(m.name).stem
        label = 0 if method == "real" else 1
        return video_item_from_member(m, sample_id=f"{method}_{vid}", label=label,
                                      method=method, group=self.pair_of.get(vid, vid.split("_")[0]),
                                      split=self.split_of.get(vid, ""))

    def _method_of(self, name: str) -> str | None:
        if "original_sequences/youtube" in name:
            return "real"
        for meth in self.methods:
            if f"manipulated_sequences/{meth}/" in name:
                return meth
        return None

    def iter_items(self):
        if self.download_script:  # stream one method at a time with the official script
            for ds in ["original"] + list(self.methods):
                with tempfile.TemporaryDirectory(dir=self.tmp_dir) as tmp:
                    cmd = [sys.executable, self.download_script, tmp, "-d", ds, "-c", self.compression,
                           "-t", "videos", "--server", self.server]
                    print("[ff++]", " ".join(cmd))
                    subprocess.run(cmd, input=b"\n" * 4, check=True)
                    with open_location(tmp) as it:
                        for m in it:
                            meth = self._method_of(m.name)
                            item = self._item(m, meth) if meth else None
                            if item:
                                yield item
            return
        for _, m in self.members():
            meth = self._method_of(m.name)
            item = self._item(m, meth) if meth else None
            if item:
                yield item


# ----------------------------------------------------------------------------- DFDC
class DFDCSource(Source):
    name = "dfdc"

    def iter_items(self):
        for loc in self.locations:
            with open_location(loc, self.tmp_dir, self.keep_downloads) as it:
                meta: dict = {}
                pending: list[Member] = []
                for m in it:
                    base = PurePosixPath(m.name).name
                    if base == "metadata.json":
                        meta.update(json.loads(m.read_bytes()))
                    elif base == "labels.csv":
                        for line in m.read_bytes().decode().splitlines()[1:]:
                            fn, lab = line.strip().split(",")[:2]
                            meta[fn] = {"label": "FAKE" if lab.strip() in ("1", "1.0") else "REAL"}
                    elif m.suffix in VIDEO_EXT:
                        pending.append(m)
                for m in pending:
                    fn = PurePosixPath(m.name).name
                    info = meta.get(fn)
                    if info is None:
                        continue
                    label = 1 if info.get("label") == "FAKE" else 0
                    group = Path(info.get("original") or fn).stem
                    yield video_item_from_member(m, sample_id=Path(fn).stem, label=label,
                                                 method="other_fake" if label else "", group=group,
                                                 split=info.get("split", "") if info.get("split") in ("train", "val", "test") else "")


# ----------------------------------------------------------------------------- Celeb-DF v2
class CelebDFSource(Source):
    name = "celebdf"

    def iter_items(self):
        for loc in self.locations:
            with open_location(loc, self.tmp_dir, self.keep_downloads) as it:
                test_set, pending = set(), []
                for m in it:
                    if m.name.endswith("List_of_testing_videos.txt"):
                        for line in m.read_bytes().decode().splitlines():
                            parts = line.strip().split()
                            if len(parts) == 2:
                                test_set.add(parts[1])
                    elif m.suffix in VIDEO_EXT:
                        pending.append(m)
                for m in pending:
                    rel = "/".join(PurePosixPath(m.name).parts[-2:])
                    folder = PurePosixPath(m.name).parent.name
                    label = 1 if folder == "Celeb-synthesis" else 0
                    stem = PurePosixPath(m.name).stem
                    group = stem.split("_")[0] if folder.startswith("Celeb") else f"yt{stem}"
                    yield video_item_from_member(m, sample_id=f"{folder}_{stem}", label=label,
                                                 method="other_fake" if label else "", group=group,
                                                 split="test" if rel in test_set else "")


# ----------------------------------------------------------------------------- DeeperForensics-1.0
class DeeperForensicsSource(Source):
    name = "deeperforensics"

    def iter_items(self):
        splits: dict[str, str] = {}
        for loc in self.locations:
            with open_location(loc, self.tmp_dir, self.keep_downloads) as it:
                pending = []
                for m in it:
                    p = PurePosixPath(m.name)
                    if p.suffix == ".txt" and p.stem in ("train", "val", "test"):
                        for line in m.read_bytes().decode().splitlines():
                            if line.strip():
                                splits[PurePosixPath(line.strip()).stem] = p.stem
                    elif m.suffix in VIDEO_EXT:
                        pending.append(m)
                for m in pending:
                    p = PurePosixPath(m.name)
                    fake = "manipulated_videos" in m.name
                    variant = p.parent.name
                    yield video_item_from_member(
                        m, sample_id=f"{variant}_{p.stem}", label=int(fake), method="other_fake" if fake else "",
                        group=p.stem.split("_")[0], split=splits.get(p.stem, ""), meta={"variant": variant})


# ----------------------------------------------------------------------------- WildDeepfake
class WildDeepfakeSource(FolderSource):
    """Archives named {real,fake}_{train,test}.tar.gz containing <video>/<seq>/<frame>.png."""

    name = "wilddeepfake"
    default_detector = "none"

    def __init__(self, **kw):
        super().__init__(images_as_sequences=True, **kw)

    def iter_items(self):
        for loc in self.locations:
            base = PurePosixPath(loc.split("?")[0]).name.lower()
            label = 1 if "fake" in base else 0
            split = "test" if "test" in base else ""
            self.rules = [dict(pattern=".*", label=label, method="other_fake" if label else "", split=split,
                               _re=re.compile(".*"))]
            saved = self.locations
            self.locations = [loc]
            self.group_re = re.compile(r"([^/]+)/[^/]+/[^/]+$")
            try:
                for item in super().iter_items():
                    item.sample_id = f"{'fake' if label else 'real'}_{item.sample_id}"
                    yield item
            finally:
                self.locations = saved


# ----------------------------------------------------------------------------- DeepfakeBench preprocessed
class DeepfakeBenchSource(Source):
    """Imports DeepfakeBench's rearranged JSON (already-cropped frames)."""

    name = "deepfakebench"
    default_detector = "none"

    def __init__(self, json_dir=None, data_root=None, datasets=None, **kw):
        super().__init__(**kw)
        self.json_dir, self.data_root = Path(json_dir), Path(data_root) if data_root else None
        self.datasets = datasets or ["UADFV"]

    def iter_items(self):
        for ds in self.datasets:
            info = json.loads((self.json_dir / f"{ds}.json").read_text())[ds]
            for label_name, splits in info.items():
                for split, videos in splits.items():
                    for vid, rec in (videos.items() if isinstance(videos, dict) else []):
                        if not isinstance(rec, dict) or "frames" not in rec:
                            # c23/c40 nesting for FF++
                            for sub_vid, sub_rec in rec.items():
                                yield self._item(ds, label_name, split, sub_vid, sub_rec)
                            continue
                        yield self._item(ds, label_name, split, vid, rec)

    def _item(self, ds, label_name, split, vid, rec):
        label = 0 if "real" in label_name.lower() else 1
        frames = [self._resolve(f) for f in rec["frames"]]
        split = {"train": "train", "val": "val", "test": "test"}.get(split, "")
        return Item(sample_id=f"{ds}_{label_name}_{vid}", label=label, kind="images",
                    fetch=lambda tmp, f=frames: f, method="other_fake" if label else "", group=vid, split=split,
                    meta={"source_dataset": ds})

    def _resolve(self, f: str) -> Path:
        p = Path(f)
        if self.data_root and not p.is_absolute():
            # DeepfakeBench stores paths like datasets/rgb/UADFV/...
            parts = p.parts
            if "rgb" in parts:
                p = Path(*parts[parts.index("rgb") + 1:])
            p = self.data_root / p
        return p


# ----------------------------------------------------------------------------- registry
def _uadfv(**kw):
    rules = [dict(pattern=r"(^|/)fake/", label=1, method="other_fake"), dict(pattern=r"(^|/)real/", label=0)]
    return FolderSource(rules=kw.pop("rules", rules), dataset_name="uadfv",
                        group_pattern=kw.pop("group_pattern", r"/(\d+)[^/]*$"), **kw)


def _vggface2(**kw):
    rules = kw.pop("rules", [dict(pattern=r"\.(jpg|png)$", label=0)])
    src = FolderSource(rules=rules, dataset_name="vggface2", group_pattern=r"(n\d{6})/",
                       max_per_group=kw.pop("max_per_identity", 50), **kw)
    return src


def _dffd(**kw):
    rules = kw.pop("rules", [
        dict(pattern=r"(ffhq|celeba)", label=0),
        dict(pattern=r"(stylegan|pggan|stargan|faceapp)", label=1, method="gan_synthesis"),
    ])
    return FolderSource(rules=rules, dataset_name="dffd", **kw)


def _generic(name):
    def f(**kw):
        return FolderSource(dataset_name=kw.pop("dataset_name", name), **kw)
    return f


SOURCES = {
    "faceforensics": FaceForensicsSource,
    "dfdc": DFDCSource,
    "celebdf": CelebDFSource,
    "deeperforensics": DeeperForensicsSource,
    "wilddeepfake": WildDeepfakeSource,
    "deepfakebench": DeepfakeBenchSource,
    "uadfv": _uadfv,
    "vggface2": _vggface2,
    "dffd": _dffd,
    "deephy": _generic("deephy"),
    "megammdf": _generic("megammdf"),
    "folder": _generic("folder"),
}


def build_source(name: str, **cfg) -> Source:
    if name not in SOURCES:
        raise KeyError(f"Unknown dataset {name!r}; available: {sorted(SOURCES)}")
    return SOURCES[name](**cfg)
