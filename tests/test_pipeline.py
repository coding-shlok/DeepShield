"""End-to-end: train -> evaluate -> KYC -> robustness -> export -> predict, plus SimCLR/KD/ensemble."""
import json
from pathlib import Path

import torch

from deepshield.utils.config import AttrDict


def test_train_eval_export_predict(tiny_cfg, tmp_path):
    from deepshield.deploy.export import export_all
    from deepshield.deploy.inference import DeepShieldPredictor
    from deepshield.evaluation.evaluate import evaluate, kyc_suite, robustness_eval
    from deepshield.evaluation.report import build_report
    from deepshield.training.trainer import train

    r = train(tiny_cfg)
    ckpt = Path(r["out_dir"]) / "best.pth"
    assert ckpt.exists() and (Path(r["out_dir"]) / "last.pth").exists()
    st = torch.load(ckpt, weights_only=False)
    assert 0 <= st["threshold"] <= 1

    cfg = AttrDict.wrap(dict(tiny_cfg))
    cfg["eval"] = AttrDict.wrap({"checkpoint": str(ckpt), "out_dir": str(tmp_path / "eval"), "in_domain": ["x"],
                                 "presets": ["clean", "jpeg_q70", "combined"], "latency_iters": 2, "max_samples": 8,
                                 "attacks": ["fgsm", "pgd"], "eps": [8 / 255], "pgd_steps": 2,
                                 "smoothing": {"sigma": 0.1, "n0": 4, "n": 10, "samples": 2, "batch": 10}})
    m = evaluate(cfg)
    assert "synthetic" in m and (tmp_path / "eval" / "roc.png").exists()
    k = kyc_suite(cfg)
    assert {"clean", "jpeg_q70", "combined", "imbalance_80_20", "latency"} <= set(k)
    rob = robustness_eval(cfg)
    assert {"clean", "fgsm_eps8", "pgd_eps8", "randomized_smoothing", "perturbation_detection"} <= set(rob)
    text = build_report([str(tmp_path / "eval")], str(tmp_path / "REPORT.md"))
    assert "PRD success criteria" in text

    cfg["export"] = AttrDict.wrap({"checkpoint": str(ckpt), "out_dir": str(tmp_path / "export"), "calib_samples": 2,
                                   "tensorrt": False})
    rep = export_all(cfg)
    assert rep["onnx"]["ok"], rep["onnx"]
    assert Path(rep["int8_onnx"]["path"]).stat().st_size < Path(rep["onnx"]["path"]).stat().st_size

    frames_dir = Path(cfg.data.manifests[0]).parent / "faces"
    img = next(frames_dir.rglob("000.jpg"))
    for model_path in (str(ckpt), rep["onnx"]["path"]):
        p = DeepShieldPredictor(model_path, detector="none")
        out = p.predict(str(img))
        assert 0 <= out.prob_fake <= 1 and out.decision in ("accept", "reject", "manual_review")
        batch = p.predict_batch([str(img), str(img)])
        assert len(batch) == 2 and abs(batch[0].prob_fake - batch[1].prob_fake) < 1e-5


def test_pretrain_distill_ensemble(tiny_cfg, tmp_path):
    from deepshield.training.distill import distill
    from deepshield.training.ensemble import fit_ensemble_cmd
    from deepshield.training.pretrain import pretrain
    from deepshield.training.trainer import train

    pcfg = AttrDict.wrap({"out_dir": str(tmp_path / "simclr"),
                          "data": {"manifests": tiny_cfg.data.manifests, "image_size": 48, "batch_size": 8, "num_workers": 0},
                          "model": {"backbone": dict(tiny_cfg.model.backbone), "simclr": {"proj_dim": 16, "hidden": 32}},
                          "train": {"epochs": 1, "max_steps": 2, "warmup_steps": 1}})
    bb = pretrain(pcfg)
    assert Path(bb).exists()

    tcfg = AttrDict.wrap(dict(tiny_cfg))
    tcfg["model"] = AttrDict.wrap(dict(tiny_cfg.model) | {"backbone": dict(tiny_cfg.model.backbone, checkpoint=bb)})
    tcfg["train"] = AttrDict.wrap(dict(tiny_cfg.train) | {"adversarial": {"enabled": False}, "max_steps": 3})
    teacher = Path(train(tcfg)["out_dir"]) / "best.pth"

    dcfg = AttrDict.wrap(dict(tiny_cfg))
    dcfg["out_dir"] = str(tmp_path / "kd")
    dcfg["model"] = AttrDict.wrap({"backbone": {"name": "efficientnet_b0", "pretrained": False, "embed_dim": 32},
                                   "temporal": "none", "fusion": {"dim": 32, "heads": 4}})
    dcfg["distill"] = AttrDict.wrap({"teacher_checkpoint": str(teacher)})
    dcfg["train"] = AttrDict.wrap({"epochs": 1, "max_steps": 3, "warmup_steps": 1})
    distill(dcfg)
    student = tmp_path / "kd" / "best.pth"
    assert student.exists()

    ecfg = AttrDict.wrap(dict(tiny_cfg))
    ecfg["out_dir"] = str(tmp_path / "ens")
    ecfg["ensemble"] = AttrDict.wrap({"mode": "gated", "epochs": 20,
                                      "members": {"teacher": {"checkpoint": str(teacher)},
                                                  "student": {"checkpoint": str(student)}}})
    rep = fit_ensemble_cmd(ecfg)
    assert "ensemble_gated" in rep
    from deepshield.models.factory import load_model

    ens = load_model(str(tmp_path / "ens" / "ensemble.pth"))
    assert ens(torch.rand(1, 3, 3, 48, 48))["logits"].shape == (1, 2)
