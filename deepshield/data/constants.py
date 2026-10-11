"""Shared label vocabularies.

Binary label: 0 = real (bona fide), 1 = fake (attack).
Method label (auxiliary multi-task head): manipulation family. Datasets that do not
expose a method use "other_fake"; unknown -> ignored in the loss.
"""

REAL, FAKE = 0, 1

METHODS = [
    "real",
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
    "gan_synthesis",   # fully synthetic faces (DFFD StyleGAN/PGGAN, ...)
    "other_fake",      # DFDC/Celeb-DF/DeeperForensics/Wild... swaps without method tag
]
METHOD_TO_IDX = {m.lower(): i for i, m in enumerate(METHODS)}
IGNORE_INDEX = -100


def method_index(method: str | None, label: int) -> int:
    if method is None or method == "" or (isinstance(method, float)):
        return METHOD_TO_IDX["real"] if label == REAL else METHOD_TO_IDX["other_fake"]
    key = str(method).lower()
    if key in METHOD_TO_IDX:
        return METHOD_TO_IDX[key]
    if key in {"unknown", "ignore"}:
        return IGNORE_INDEX
    return METHOD_TO_IDX["real"] if label == REAL else METHOD_TO_IDX["other_fake"]


MANIFEST_COLUMNS = [
    "sample_id",   # unique id within dataset (video id or image id)
    "dataset",     # dataset name
    "label",       # 0 real / 1 fake
    "method",      # manipulation method (see METHODS) or empty
    "group",       # identity / source-video id used for leakage-free splitting
    "path",        # directory with face crops (relative to manifest dir) or single image path
    "n_frames",    # number of face crops stored
    "split",       # train / val / test
]
