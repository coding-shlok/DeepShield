"""Evaluation entry points: benchmark / cross-dataset (D2), KYC suite (D3), adversarial (D4)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.augment import KYC_EVAL_PRESETS
from ..data.datasets import build_dataloader, build_dataset, subsample_ratio
from ..models.factory import load_model
from ..utils.distributed import init_distributed, is_main_process
from ..utils.logger import get_logger
from ..training.engine import run_inference
from .metrics import PRD_TARGETS, compute_metrics, generalization_gap
from .plots import plot_attack, plot_bars, plot_confusion, plot_roc

log = get_logger()


def _threshold(model, ec) -> float:
    """`eval.threshold: auto` (default) uses the validation-EER threshold stored in the checkpoint."""
    t = ec.get("threshold", "auto")
    return float(getattr(model, "calibrated_threshold", 0.5)) if t in (None, "auto") else float(t)


def _clean(m: dict) -> dict:
    return {k: v for k, v in m.items() if not isinstance(v, list)} | {"confusion": m.get("confusion")}


def _predict(model, data_cfg, split, device, preset=None, attack=None, max_samples=None, desc="eval"):
    dc = dict(data_cfg)
    if max_samples:
        dc[f"max_{split}_samples"] = max_samples
    ds = build_dataset(dc, split, train=False, preset=preset)
    dl = build_dataloader(ds, dc | {"batch_size": dc.get("eval_batch_size", dc.get("batch_size", 16))}, train=False)
    res = run_inference(model, dl, device, amp=True, attack=attack, desc=desc)
    res["dataset"] = np.array([ds.dataset_names[i] for i in res["dataset_idx"]])
    res["sample_id"] = ds.df["sample_id"].to_numpy()[res["index"]]
    return ds, res


def _per_dataset(res, threshold=0.5) -> dict:
    out = {"all": compute_metrics(res["labels"], res["probs"], threshold)}
    for name in np.unique(res["dataset"]):
        m = res["dataset"] == name
        out[str(name)] = compute_metrics(res["labels"][m], res["probs"][m], threshold)
    return out


def _save_preds(res, path):
    pd.DataFrame({"sample_id": res["sample_id"], "dataset": res["dataset"], "label": res["labels"],
                  "prob_fake": res["probs"]}).to_csv(path, index=False)


# ----------------------------------------------------------------------------- D2 benchmark / cross-dataset
def evaluate(cfg) -> dict:
    """Per-dataset metrics on the test split + ROC/confusion plots + generalisation gap.

    `eval.in_domain` lists datasets seen during training; others count as unseen."""
    device = init_distributed()
    ec = cfg.get("eval", {})
    out = Path(ec.get("out_dir", "outputs/eval"))
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(ec["checkpoint"]).to(device)
    threshold = _threshold(model, ec)
    log.info(f"decision threshold = {threshold:.4f}")
    _, res = _predict(model, cfg["data"], ec.get("split", "test"), device, desc="test")
    if not is_main_process():
        return {}
    metrics = _per_dataset(res, threshold)
    _save_preds(res, out / "predictions.csv")
    curves = {n: (res["labels"][res["dataset"] == n], res["probs"][res["dataset"] == n]) for n in np.unique(res["dataset"])}
    plot_roc(curves, out / "roc.png", "ROC per dataset")
    for name, m in metrics.items():
        plot_confusion(m["confusion"], out / f"confusion_{name}.png", f"{name} (thr={threshold})")

    in_domain = ec.get("in_domain") or []
    seen = [n for n in metrics if n in in_domain]
    unseen = [n for n in metrics if n not in in_domain and n != "all"]
    if seen and unseen:
        acc_seen = float(np.mean([metrics[n]["acc"] for n in seen]))
        acc_unseen = float(np.mean([metrics[n]["acc"] for n in unseen]))
        metrics["generalization"] = {"acc_in_domain": acc_seen, "acc_unseen": acc_unseen,
                                     "gap": generalization_gap(acc_seen, acc_unseen), "seen": seen, "unseen": unseen}
    (out / "metrics.json").write_text(json.dumps({k: _clean(v) if "confusion" in v else v for k, v in metrics.items()},
                                                 indent=2))
    _print_table(metrics, ["acc", "auc", "eer", "f1_macro", "apcer", "bpcer"])
    return metrics


def _print_table(metrics: dict, keys):
    rows = {n: {k: m.get(k) for k in keys} for n, m in metrics.items() if isinstance(m, dict) and "acc" in m}
    log.info("\n" + pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))


# ----------------------------------------------------------------------------- D3 KYC suite
def kyc_suite(cfg) -> dict:
    device = init_distributed()
    ec = cfg.get("eval", {})
    out = Path(ec.get("out_dir", "outputs/kyc"))
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(ec["checkpoint"]).to(device)
    presets = ec.get("presets") or list(KYC_EVAL_PRESETS)
    thr = _threshold(model, ec)
    split = ec.get("split", "test")
    results, clean_res = {}, None
    for p in presets:
        _, res = _predict(model, cfg["data"], split, device, preset=None if p == "clean" else p,
                          max_samples=ec.get("max_samples"), desc=p)
        if p == "clean":
            clean_res = res
        results[p] = _clean(compute_metrics(res["labels"], res["probs"], thr))
        log.info(f"[kyc] {p:22s} acc={results[p]['acc']:.4f} auc={results[p]['auc']:.4f} eer={results[p]['eer']:.4f}")

    # class imbalance 80:20 (bootstrap over subsamples of the clean predictions)
    if clean_res is not None:
        df = pd.DataFrame({"label": clean_res["labels"], "prob": clean_res["probs"]})
        reps = []
        for s in range(ec.get("imbalance_repeats", 20)):
            sub = subsample_ratio(df, 0.8, seed=s)
            if len(sub) and sub.label.nunique() == 2:
                reps.append(compute_metrics(sub["label"], sub["prob"], thr))
        if reps:
            keys = ["acc", "auc", "f1_macro", "f1_fake", "f1_real", "apcer", "bpcer", "eer"]
            results["imbalance_80_20"] = {k: float(np.mean([r[k] for r in reps])) for k in keys} | \
                {f"{k}_std": float(np.std([r[k] for r in reps])) for k in keys} | {"n_repeats": len(reps)}

    if ec.get("latency", True):
        from ..deploy.profile import profile_model

        lat = {}
        lat["cpu"] = profile_model(model.cpu(), device="cpu", clip_len=cfg["data"].get("clip_len", 16),
                                   image_size=cfg["data"].get("image_size", 224), iters=ec.get("latency_iters", 10))
        if torch.cuda.is_available():
            lat["gpu"] = profile_model(model.cuda(), device="cuda", clip_len=cfg["data"].get("clip_len", 16),
                                       image_size=cfg["data"].get("image_size", 224), iters=ec.get("latency_iters", 50))
        results["latency"] = lat
    if is_main_process():
        accs = {k: v["acc"] for k, v in results.items() if isinstance(v, dict) and "acc" in v and k in presets}
        plot_bars(accs, out / "kyc_accuracy.png", "accuracy", "Accuracy under KYC capture conditions", 0.85)
        (out / "kyc_results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


# ----------------------------------------------------------------------------- D4 adversarial robustness
def robustness_eval(cfg) -> dict:
    from ..robustness import ATTACKS, FeatureSqueezingDetector, Smooth
    from ..robustness.smoothing import certified_accuracy

    device = init_distributed()
    ec = cfg.get("eval", {})
    out = Path(ec.get("out_dir", "outputs/robustness"))
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(ec["checkpoint"]).to(device).eval()
    split, nmax = ec.get("split", "test"), ec.get("max_samples", 512)
    thr = _threshold(model, ec)
    results = {}
    _, clean = _predict(model, cfg["data"], split, device, max_samples=nmax, desc="clean")
    results["clean"] = _clean(compute_metrics(clean["labels"], clean["probs"], thr))
    for atk in ec.get("attacks", ["fgsm", "pgd", "autoattack"]):
        for eps in ec.get("eps", [2 / 255, 4 / 255, 8 / 255]):
            kw = {"steps": ec.get("pgd_steps", 20)} if atk == "pgd" else {}
            fn = (lambda m, x, y, a=ATTACKS[atk], e=eps, k=kw: a(m, x, y, eps=e, **k))
            _, r = _predict(model, cfg["data"], split, device, attack=fn, max_samples=nmax, desc=f"{atk}@{eps:.4f}")
            key = f"{atk}_eps{round(eps * 255)}"
            results[key] = _clean(compute_metrics(r["labels"], r["probs"], thr))
            log.info(f"[robust] {key:18s} acc={results[key]['acc']:.4f} auc={results[key]['auc']:.4f}")

    # randomized smoothing certification on a small subset (expensive: n Monte-Carlo samples each)
    sc = ec.get("smoothing", {"sigma": 0.12, "n": 200, "n0": 50, "samples": 50})
    if sc:
        ds = build_dataset(cfg["data"], split, train=False)
        idx = np.random.default_rng(0).choice(len(ds), min(sc.get("samples", 50), len(ds)), replace=False)
        smooth = Smooth(model, 2, sc.get("sigma", 0.12), batch=sc.get("batch", 16))
        preds, radii, labels = [], [], []
        for i in idx:
            item = ds[int(i)]
            c, r = smooth.certify(item["frames"].to(device), sc.get("n0", 50), sc.get("n", 200), sc.get("alpha", 0.001))
            preds.append(c), radii.append(r), labels.append(int(item["label"]))
        results["randomized_smoothing"] = {"sigma": sc.get("sigma", 0.12), "abstain_rate": float(np.mean(np.array(preds) == -1)),
                                           **certified_accuracy(np.array(preds), np.array(radii), np.array(labels))}
        log.info(f"[robust] smoothing {results['randomized_smoothing']}")

    # perturbation detection (feature squeezing) + visualisation on one batch
    ds = build_dataset(cfg["data"] | {"max_val_samples": nmax}, "val", train=False)
    dl = build_dataloader(ds, cfg["data"] | {"num_workers": 2}, train=False)
    det = FeatureSqueezingDetector(model)
    clean_scores, adv_scores, vis_done = [], [], False
    eps = ec.get("detect_eps", 8 / 255)
    for b in dl:
        x, y = b["frames"].to(device), b["label"].to(device)
        x_adv = ATTACKS["pgd"](model, x, y, eps=eps, steps=10)
        clean_scores.append(det.score(x).cpu())
        adv_scores.append(det.score(x_adv).cpu())
        if not vis_done:
            xg = x.clone().requires_grad_(True)
            o = model(xg)["logits"]
            o[:, 1].sum().backward()
            sal = xg.grad.abs().amax(-3)  # [B,(T),H,W]
            sel = (lambda t: t[:, 0] if t.dim() == 5 else t)
            with torch.no_grad():
                pc = model(x)["logits"].softmax(-1)[:, 1].cpu().numpy()
                pa = model(x_adv)["logits"].softmax(-1)[:, 1].cpu().numpy()
            s = sal[:, 0] if sal.dim() == 4 else sal
            plot_attack(sel(x).detach().cpu().numpy(), sel(x_adv).cpu().numpy(), s.cpu().numpy(), pc, pa,
                        out / "attack_visualization.png")
            vis_done = True
        if sum(len(c) for c in clean_scores) >= nmax:
            break
    cs, as_ = torch.cat(clean_scores).numpy(), torch.cat(adv_scores).numpy()
    thr = det.calibrate(cs, fpr=0.05)
    results["perturbation_detection"] = {"threshold": thr, "fpr_clean": float((cs > thr).mean()),
                                         "detection_rate_pgd": float((as_ > thr).mean())}
    if is_main_process():
        accs = {k: v["acc"] for k, v in results.items() if isinstance(v, dict) and "acc" in v}
        plot_bars(accs, out / "adversarial_accuracy.png", "accuracy", "Accuracy under attack", 0.85)
        (out / "robustness_results.json").write_text(json.dumps(results, indent=2, default=float))
    return results
