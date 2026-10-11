# Reproduction guide

1. **Environment.** Run `bash jobs/setup_env.sh` (Python 3.10, PyTorch ≥ 2.1, CUDA wheels). Record `pip freeze > outputs/env.txt`.
2. **Seeds.** `seed: 42` in `configs/base.yaml` seeds Python, NumPy, torch and the dataloader workers. Set `deterministic: true` for bit-exact cuDNN, which is slower. Splits are seeded separately (`seed` in the data configs).
3. **Configs.** Every run writes its fully resolved `config.yaml` into `runs/<name>/`, and every checkpoint embeds that config. `load_model(ckpt)` rebuilds the exact architecture.
4. **Data.**
   ```bash
   for d in celebdf faceforensics dfdc deeperforensics wilddeepfake uadfv vggface2 dffd; do
     qsub -v DATA=$d jobs/pbs/prepare.pbs
   done
   ```
   Then run `deepshield datasets check data.manifests="[...]" verify_images=true` and archive the `manifest.csv` files. They define the splits.
5. **Training.** Run `bash jobs/run_pipeline.sh`. Stage outputs:
   `runs/stage1_simclr/backbone.pth` → `runs/stage2_resnet_bilstm/best.pth` → `runs/stage3_robust/best.pth` →
   `runs/effnet_distill/best.pth`, `runs/vit_b16/best.pth` → `runs/ensemble/ensemble.pth`.
6. **Evaluation.** Run `qsub -v CKPT=runs/stage3_robust/best.pth,TAG=stage3 jobs/pbs/evaluate.pbs`. It writes `outputs/stage3/REPORT.md` with every table, figure and PRD target check.
7. **Ablations** (PRD Contribution 1–3). Use config overrides, for example:
   * single-dataset vs multi-stage: `data.manifests="[data/processed/faceforensics/manifest.csv]"`
   * no SimCLR: `model.backbone.checkpoint=null`
   * no adversarial training: `train.adversarial.enabled=false`
   * focal vs CE: `loss.focal.gamma=0 loss.focal.alpha=null`
   * natural 80:20 sampling: `data.balanced_sampling=false loss.focal.alpha="[0.25,0.75]"`
   * 8-frame clips for latency (PRD §9): `data.clip_len=8`
   * ensemble modes: `ensemble.mode=mean|weighted|confidence|gated`

## Hyper-parameters (PRD §6.3)

AdamW, lr 1e-4 (backbone ×0.5), weight decay 1e-5, 1k warm-up steps then cosine decay, gradient clipping at norm 1.0, bf16 AMP. Focal γ=2. Auxiliary weights: method 0.2, frame 0.3, temporal 0.1, temporal-score 0.1, contrastive 0.05. Stage 3: PGD-3 at ε=8/255 on 50 % of batches (loss weight 0.5), mixup p=0.3 α=0.4, Gaussian noise σ=0.12 on 30 % of batches.
