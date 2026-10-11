#!/usr/bin/env bash
# End-to-end CPU smoke test on synthetic data (~3 min): data -> train -> eval -> KYC -> robustness -> export -> predict.
set -euo pipefail
cd "$(dirname "$0")/.."
deepshield synthetic out_dir=data/processed/synthetic n_videos=160 n_frames=12 size=96
deepshield train       -c configs/experiments/smoke.yaml
deepshield evaluate    -c configs/experiments/smoke.yaml --out outputs/smoke/benchmark
deepshield kyc-eval    -c configs/experiments/smoke.yaml --out outputs/smoke/kyc
deepshield robust-eval -c configs/experiments/smoke.yaml --out outputs/smoke/robustness
deepshield export      -c configs/experiments/smoke.yaml export.out_dir=outputs/smoke/export
deepshield predict --checkpoint outputs/smoke/export/deepshield.onnx \
  --inputs data/processed/synthetic/faces/synthetic_00000/000.jpg detector=none
deepshield report --dirs outputs/smoke/benchmark outputs/smoke/kyc outputs/smoke/robustness outputs/smoke/export \
  --out outputs/smoke/REPORT.md
echo "done -> outputs/smoke/REPORT.md"
