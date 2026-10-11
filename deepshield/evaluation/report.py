"""Aggregate evaluation JSONs into a Markdown benchmark report with PRD target checks (D2-D4)."""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import PRD_TARGETS, check_target


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _mark(ok):
    return "PASS" if ok else "FAIL"


def build_report(eval_dirs: list[str], out_path: str = "outputs/REPORT.md") -> str:
    import os

    rel = (lambda p: Path(os.path.relpath(p, Path(out_path).resolve().parent)).as_posix())
    lines = ["# DeepShield benchmark report", ""]
    checks = []
    for d in eval_dirs:
        d = Path(d)
        if (d / "metrics.json").exists():
            m = json.loads((d / "metrics.json").read_text())
            lines += [f"## Benchmark / cross-dataset ({d})", "",
                      "| dataset | n | acc | AUC | EER | macro-F1 | APCER | BPCER |", "|---|---|---|---|---|---|---|---|"]
            for name, r in m.items():
                if isinstance(r, dict) and "acc" in r:
                    lines.append(f"| {name} | {r['n']} | {_fmt(r['acc'])} | {_fmt(r['auc'])} | {_fmt(r['eer'])} | "
                                 f"{_fmt(r['f1_macro'])} | {_fmt(r['apcer'])} | {_fmt(r['bpcer'])} |")
                    for t_name, targets in PRD_TARGETS.items():
                        if t_name == name:
                            for k, (op, val) in targets.items():
                                checks.append((f"{name} {k} {op} {val}", r[k], check_target(r[k], op, val)))
            if "generalization" in m:
                g = m["generalization"]
                lines += ["", f"Generalisation gap: **{g['gap']:.4f}** (in-domain {g['acc_in_domain']:.4f} vs unseen "
                          f"{g['acc_unseen']:.4f})"]
                checks.append(("generalization gap <= 0.10", g["gap"], check_target(g["gap"], "<=", 0.10)))
            lines += ["", f"![ROC]({rel((d / 'roc.png').resolve())})", ""]
        if (d / "kyc_results.json").exists():
            k = json.loads((d / "kyc_results.json").read_text())
            lines += [f"## KYC evaluation suite ({d})", "", "| condition | acc | AUC | EER | BPCER |", "|---|---|---|---|---|"]
            for name, r in k.items():
                if isinstance(r, dict) and "acc" in r and "acc_std" not in r:
                    lines.append(f"| {name} | {_fmt(r['acc'])} | {_fmt(r['auc'])} | {_fmt(r['eer'])} | {_fmt(r['bpcer'])} |")
            if "combined" in k:
                checks.append(("KYC combined perturbations acc >= 0.85", k["combined"]["acc"], k["combined"]["acc"] >= 0.85))
            if "imbalance_80_20" in k:
                im = k["imbalance_80_20"]
                lines += ["", f"80:20 imbalance: macro-F1 {im['f1_macro']:.4f}±{im['f1_macro_std']:.4f}, "
                          f"BPCER {im['bpcer']:.4f}, APCER {im['apcer']:.4f}"]
                checks.append(("80:20 balanced F1 >= 0.92", im["f1_macro"], im["f1_macro"] >= 0.92))
                checks.append(("80:20 false rejection < 3%", im["bpcer"], im["bpcer"] < 0.03))
            if "latency" in k:
                lines += ["", "| device | total median ms | size MB |", "|---|---|---|"]
                for dev, r in k["latency"].items():
                    lines.append(f"| {dev} | {r['ms']:.1f} | {r['size_mb_fp32']:.1f} |")
                    lim = 100 if dev == "gpu" else 300
                    checks.append((f"latency {dev} < {lim} ms", r["ms"], r["ms"] < lim))
            lines += ["", f"![KYC]({rel((d / 'kyc_accuracy.png').resolve())})", ""]
        if (d / "robustness_results.json").exists():
            r = json.loads((d / "robustness_results.json").read_text())
            lines += [f"## Adversarial robustness ({d})", "", "| setting | acc | AUC |", "|---|---|---|"]
            for name, v in r.items():
                if isinstance(v, dict) and "acc" in v:
                    lines.append(f"| {name} | {_fmt(v['acc'])} | {_fmt(v['auc'])} |")
            if "pgd_eps8" in r:
                checks.append(("PGD eps=8/255 acc >= 0.85", r["pgd_eps8"]["acc"], r["pgd_eps8"]["acc"] >= 0.85))
            for key in ("randomized_smoothing", "perturbation_detection"):
                if key in r:
                    lines += ["", f"**{key}**: " + ", ".join(f"{a}={_fmt(b)}" for a, b in r[key].items())]
            lines += ["", f"![attacks]({rel((d / 'attack_visualization.png').resolve())})", ""]
        if (d / "export_report.json").exists():
            e = json.loads((d / "export_report.json").read_text())
            lines += [f"## Export ({d})", "", "```json", json.dumps(e, indent=2), "```", ""]
    if checks:
        lines += ["## PRD success criteria", "", "| criterion | value | status |", "|---|---|---|"]
        lines += [f"| {c} | {_fmt(float(v))} | {_mark(ok)} |" for c, v, ok in checks]
    text = "\n".join(lines) + "\n"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    return text
