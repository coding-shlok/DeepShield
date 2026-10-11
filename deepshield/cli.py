"""DeepShield command line.

    deepshield <command> --config <yaml> [key=value overrides ...]

Commands
  datasets info|check     dataset access instructions / processed-data integrity check
  prepare                 stream + face-crop a dataset into data/processed/<name>
  synthetic               generate a small synthetic dataset (smoke tests)
  pretrain                stage 1 SimCLR pretraining (VGGFace2 / DFFD)
  train                   stage 2/3 supervised training (DDP via torchrun)
  distill                 EfficientNet-B2 student from ResNet-50 teacher
  fit-ensemble            fit adaptive ensemble on validation logits
  evaluate                benchmark + cross-dataset metrics (D2)
  kyc-eval                KYC evaluation suite (D3)
  robust-eval             adversarial robustness analysis (D4)
  export                  ONNX / INT8 / TensorRT export (D1/D5)
  profile                 latency profiling
  predict                 run the inference pipeline on images/videos
  report                  aggregate results into a Markdown report
"""
from __future__ import annotations

import argparse
import json
import sys

from .utils.config import AttrDict, load_config


def _cfg(args) -> AttrDict:
    if args.config:
        return load_config(args.config, args.overrides)
    from .utils.config import apply_overrides

    return AttrDict.wrap(apply_overrides({}, args.overrides))


def main(argv=None):
    p = argparse.ArgumentParser(prog="deepshield", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command")
    p.add_argument("subcommand", nargs="?", default=None)
    p.add_argument("--config", "-c")
    p.add_argument("--checkpoint")
    p.add_argument("--inputs", nargs="*", default=[])
    p.add_argument("--out")
    p.add_argument("--dirs", nargs="*", default=[])
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    args, unknown = p.parse_known_args(argv)
    # allow overrides to appear anywhere
    # `--inputs a b key=value` : argparse puts overrides into inputs – move them back
    moved = [i for i in args.inputs if "=" in i and not __import__("os").path.exists(i)]
    args.inputs = [i for i in args.inputs if i not in moved]
    args.overrides = [o for o in (args.overrides + unknown + moved) if "=" in o]
    if args.subcommand and "=" in args.subcommand:
        args.overrides.insert(0, args.subcommand)
        args.subcommand = None
    cmd = args.command

    if cmd == "datasets":
        from .data.sources import DATASET_INFO

        if args.subcommand in (None, "info"):
            for name, info in DATASET_INFO.items():
                print(f"[{name}] (stage {info['stage']})\n    {info['access']}\n")
        elif args.subcommand == "check":
            from .data.prepare import check_processed

            cfg = _cfg(args)
            manifests = cfg.get("data", {}).get("manifests") or args.inputs
            ok = True
            for m in manifests:
                rep = check_processed(m, verify_images=cfg.get("verify_images", False))
                ok &= rep["ok"]
                print(m, json.dumps(rep, indent=2, default=str))
            sys.exit(0 if ok else 1)
        return
    if cmd == "prepare":
        from .data.prepare import prepare

        prepare(_cfg(args))
    elif cmd == "synthetic":
        from .data.synthetic import make_synthetic_dataset

        cfg = _cfg(args)
        path = make_synthetic_dataset(cfg.get("out_dir", "data/processed/synthetic"), cfg.get("n_videos", 120),
                                      cfg.get("n_frames", 16), cfg.get("size", 128), seed=cfg.get("seed", 0),
                                      name=cfg.get("name", "synthetic"))
        print(f"wrote {path}")
    elif cmd == "pretrain":
        from .training.pretrain import pretrain

        print(pretrain(_cfg(args)))
    elif cmd == "train":
        from .training.trainer import train

        r = train(_cfg(args))
        print(json.dumps({k: v for k, v in r.items() if k != "history"}))
    elif cmd == "distill":
        from .training.distill import distill

        print(distill(_cfg(args)))
    elif cmd == "fit-ensemble":
        from .training.ensemble import fit_ensemble_cmd

        fit_ensemble_cmd(_cfg(args))
    elif cmd in ("evaluate", "kyc-eval", "robust-eval"):
        from .evaluation import evaluate as ev

        cfg = _cfg(args)
        if args.checkpoint:
            cfg.setdefault("eval", AttrDict())["checkpoint"] = args.checkpoint
        if args.out:
            cfg.setdefault("eval", AttrDict())["out_dir"] = args.out
        {"evaluate": ev.evaluate, "kyc-eval": ev.kyc_suite, "robust-eval": ev.robustness_eval}[cmd](cfg)
    elif cmd == "export":
        from .deploy.export import export_all

        cfg = _cfg(args)
        if args.checkpoint:
            cfg.setdefault("export", AttrDict())["checkpoint"] = args.checkpoint
        export_all(cfg)
    elif cmd == "profile":
        import torch

        from .deploy.profile import profile_model
        from .models.factory import load_model

        cfg = _cfg(args)
        model = load_model(args.checkpoint)
        dc = cfg.get("data", {})
        for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
            r = profile_model(model, dev, dc.get("clip_len", 16), dc.get("image_size", 224),
                              iters=cfg.get("iters", 30))
            print(json.dumps(r, indent=2))
    elif cmd == "predict":
        from .deploy.inference import DeepShieldPredictor

        cfg = _cfg(args)
        pred = DeepShieldPredictor(args.checkpoint, clip_len=cfg.get("clip_len"), image_size=cfg.get("image_size"), detector=cfg.get("detector", "retinaface"),
                                   threshold=cfg.get("threshold"), robustness_check=cfg.get("robustness_check", False))
        for r in pred.predict_batch(args.inputs) if len(args.inputs) > 1 else [pred.predict(args.inputs[0])]:
            d = r.to_dict()
            d.pop("frame_probs", None)
            print(json.dumps(d))
    elif cmd == "report":
        from .evaluation.report import build_report

        print(build_report(args.dirs, args.out or "outputs/REPORT.md"))
    else:
        p.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
