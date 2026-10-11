"""Export & optimisation (PRD §6.4, Phase 5, D1/D5):
  * ONNX export (dynamic batch, opset 17) + numerical parity check with ONNX Runtime
  * INT8: ONNX Runtime static QDQ quantisation calibrated on real face crops (CPU / TensorRT-ready)
          and PyTorch dynamic quantisation (Linear/LSTM) as a pure-torch alternative
  * TensorRT engine build via `trtexec` (FP16 / INT8) when available
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..models.detector import ProbOutput


def export_onnx(model: torch.nn.Module, path: str | Path, clip_len: int = 16, image_size: int = 224,
                opset: int = 17, dynamic_batch: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = ProbOutput(model.eval().cpu()).eval()
    x = torch.rand(1, clip_len, 3, image_size, image_size)
    dyn = {"frames": {0: "batch"}, "prob_fake": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    kw = dict(input_names=["frames"], output_names=["prob_fake", "logits"], dynamic_axes=dyn, opset_version=opset,
              do_constant_folding=True)
    try:
        torch.onnx.export(wrapped, (x,), str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the `dynamo` kwarg
        torch.onnx.export(wrapped, (x,), str(path), **kw)
    return path


def verify_onnx(model, onnx_path, clip_len=16, image_size=224, atol=1e-3) -> dict:
    import onnxruntime as ort

    x = torch.rand(2, clip_len, 3, image_size, image_size)
    with torch.no_grad():
        ref = ProbOutput(model.eval().cpu())(x)[0].numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["prob_fake"], {"frames": x.numpy()})[0]
    err = float(np.abs(ref - got).max())
    return {"max_abs_err": err, "ok": err < atol}


class _CalibReader:
    def __init__(self, batches):
        self.it = iter(batches)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"frames": b}


def quantize_onnx_int8(onnx_path, out_path, calib_batches: list[np.ndarray], per_channel: bool = True) -> Path:
    """Static INT8 (QDQ) quantisation – ResNet-50 weights 102MB -> ~25MB."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = Path(out_path).with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(onnx_path), str(pre), skip_symbolic_shape=True)
        src = pre
    except Exception as e:
        print(f"[quant] pre-processing skipped ({e})")
        src = onnx_path
    quantize_static(str(src), str(out_path), _CalibReader(calib_batches), quant_format=QuantFormat.QDQ,
                    per_channel=per_channel, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    if pre.exists():
        pre.unlink()
    return Path(out_path)


def quantize_dynamic_torch(model: torch.nn.Module) -> torch.nn.Module:
    """PyTorch dynamic INT8 for Linear/LSTM layers (CPU). Conv layers stay FP32."""
    qmod = getattr(torch, "ao", torch).quantization
    return qmod.quantize_dynamic(model.eval().cpu(), {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8)


def build_tensorrt(onnx_path, engine_path, fp16=True, int8=False, clip_len=16, image_size=224, max_batch=8) -> Path | None:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        print("[tensorrt] trtexec not found – install TensorRT on the GPU node; skipping")
        return None
    shp = f"{clip_len}x3x{image_size}x{image_size}"
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
           f"--minShapes=frames:1x{shp}", f"--optShapes=frames:1x{shp}", f"--maxShapes=frames:{max_batch}x{shp}"]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")  # QDQ ONNX carries its own scales
    print("[tensorrt]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return Path(engine_path)


def benchmark_onnx(onnx_path, clip_len=16, image_size=224, iters=20, provider="CPUExecutionProvider") -> dict:
    import time

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=[provider])
    x = np.random.rand(1, clip_len, 3, image_size, image_size).astype(np.float32)
    for _ in range(3):
        sess.run(None, {"frames": x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, {"frames": x})
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "size_mb": Path(onnx_path).stat().st_size / 1e6, "provider": provider}


def export_all(cfg) -> dict:
    """CLI driver: ONNX + verify + INT8 (+ TensorRT) + benchmark -> export_report.json."""
    from ..data.datasets import build_dataloader, build_dataset
    from ..models.factory import load_model

    ec = cfg["export"]
    out = Path(ec.get("out_dir", "outputs/export"))
    out.mkdir(parents=True, exist_ok=True)
    T, S = cfg["data"].get("clip_len", 16), cfg["data"].get("image_size", 224)
    model = load_model(ec["checkpoint"])
    report = {}
    onnx_path = export_onnx(model, out / "deepshield.onnx", T, S)
    meta = {"threshold": float(getattr(model, "calibrated_threshold", 0.5)), "clip_len": T, "image_size": S,
            "input": "frames [B,T,3,H,W] float32 RGB in [0,1]", "outputs": ["prob_fake [B]", "logits [B,2]"]}
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    report["onnx"] = {"path": str(onnx_path), **verify_onnx(model, onnx_path, T, S)}
    report["onnx"]["bench_cpu"] = benchmark_onnx(onnx_path, T, S)
    if ec.get("int8", True):
        ds = build_dataset(cfg["data"] | {"max_val_samples": ec.get("calib_samples", 64)}, "val", train=False)
        dl = build_dataloader(ds, cfg["data"] | {"batch_size": 1, "num_workers": 2}, train=False)
        calib = [b["frames"].numpy() for _, b in zip(range(ec.get("calib_samples", 64)), dl)]
        q = quantize_onnx_int8(onnx_path, out / "deepshield_int8.onnx", calib)
        q.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        report["int8_onnx"] = {"path": str(q), "bench_cpu": benchmark_onnx(q, T, S)}
        qt = quantize_dynamic_torch(load_model(ec["checkpoint"]))
        torch.save(qt.state_dict(), out / "deepshield_dynamic_int8.pth")
        report["int8_torch_dynamic"] = {"path": str(out / "deepshield_dynamic_int8.pth"),
                                        "size_mb": (out / "deepshield_dynamic_int8.pth").stat().st_size / 1e6}
    if ec.get("tensorrt", True):
        eng = build_tensorrt(onnx_path, out / "deepshield_fp16.engine", True, False, T, S)
        report["tensorrt"] = {"path": str(eng) if eng else None}
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report
