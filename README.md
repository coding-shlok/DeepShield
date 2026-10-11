# DeepShield

**Robust deepfake detection for secure biometric KYC authentication.** This is the implementation of the DeepShield PRD (v1.0).

DeepShield works out whether a selfie image or a short liveness video shows a real person or a synthetic or manipulated face. It is built for real KYC conditions: low-resolution phone cameras, motion blur, messenger JPEG compression, bad lighting, occlusions, an 80:20 real:fake prior, and adversarial attackers.

```
 video / selfie ──► Input module ──► Spatial extractor ──► Temporal module ──► Attention fusion ──► P(fake)
                   16 keyframes,     ResNet-50 multi-     2-layer BiLSTM       4-head self-attn     + method head
                   face crop 224,    scale pyramid +      (hidden 512)         over spatial +       + frame scores
                   [0,1]→[-1,1]      SRM noise stream     temporal score       temporal tokens      + robustness filter
                                     (2048-d)
 ensemble members: EfficientNet-B2 (distilled, mobile) · ViT-B/16 · R(2+1)D 3D-CNN → adaptive gated voting
```

## Contents

| Path | What |
|---|---|
| `deepshield/data/` | streaming dataset adapters for 11 datasets, RetinaFace cropping, KYC perturbation simulator, leakage-free 70/15/15 splits, samplers |
| `deepshield/models/` | `DeepShieldNet` (ResNet-50 + BiLSTM + attention, multi-task heads), EfficientNet/ViT variants, 3D-CNN, adaptive ensemble, SimCLR |
| `deepshield/losses.py` | focal loss (γ=2), method CE, temporal consistency, contrastive margin, distillation |
| `deepshield/training/` | DDP trainer (AMP, warm-up + cosine, grad clip, mixup, FGSM/PGD adversarial training, auto-resume), SimCLR pretraining, KD, ensemble fitting |
| `deepshield/robustness/` | FGSM, PGD, APGD-CE, Square, AutoAttack; randomized smoothing certification; feature-squeezing perturbation detector |
| `deepshield/evaluation/` | Acc/AUC/EER/APCER/BPCER/F1, generalisation gap, KYC suite, cross-dataset, plots, Markdown report with PRD pass/fail |
| `deepshield/deploy/` | ONNX export + parity check, INT8 (ONNX Runtime QDQ, torch dynamic), TensorRT, latency profiler, `DeepShieldPredictor` |
| `configs/` | `data/*.yaml` per dataset, `experiments/*.yaml` per training stage |
| `jobs/` | PBS Pro (Altair) job scripts + a dependency-chained pipeline launcher |
| `docs/` | data pipeline, reproduction guide, PRD traceability matrix |

## 1. Setup

```bash
git clone <repo> && cd DeepShield
bash jobs/setup_env.sh                 # conda env + torch (CUDA 12.1 wheels) + deps + `pip install -e .`
# or manually: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#              pip install -r requirements.txt && pip install -e .
```

Check that everything works without any real data (CPU, about 3 minutes):

```bash
bash scripts/smoke_test.sh            # synthetic data -> train -> all evaluations -> ONNX/INT8 -> report
pytest -q                             # 19 unit/integration tests
```

## 2. Data: stream, don't store

Every PRD dataset is gated behind a licence or request form, so nothing can be downloaded anonymously. Once you have access, paste the link or credentials into `configs/data/<name>.yaml` and run `deepshield prepare`. The raw data is **streamed**:

* **remote `.zip`** (DFDC parts, Celeb-DF, …) is read with HTTP Range requests. Only the bytes of the video being processed are fetched, so a 10 GB DFDC part never touches the disk.
* **remote `.tar.gz`** (WildDeepfake, VGGFace2 mirrors) is decoded as a stream.
* **FF++** runs the official download script one method at a time into a temp dir.
* Everything else (Google Drive, Kaggle, servers without Range support) is downloaded to a temp file and deleted as soon as it has been processed.

Only the face crops are kept: 32 crops × 256² JPEG per video, about 0.3 MB, so the whole of DFDC is roughly 40 GB instead of 470 GB. Preparation is resumable: re-run the same command after an interruption.

```bash
deepshield datasets info                              # how to get access to each dataset
deepshield prepare -c configs/data/celebdf.yaml       # or: qsub -v DATA=celebdf jobs/pbs/prepare.pbs
deepshield prepare -c configs/data/dfdc.yaml limit=2000   # quick subset first
deepshield datasets check data.manifests="[data/processed/celebdf/manifest.csv]" verify_images=true
```

Recommended order, given access delays (PRD §9): Celeb-DF and FF++ c23 (small, fast), then DFDC parts, then DeeperForensics and WildDeepfake, then VGGFace2 and DFFD. Details are in [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).

## 3. Training pipeline (PRD §6.2)

| Stage | Config | Data | What |
|---|---|---|---|
| 1 | `stage1_simclr.yaml` | VGGFace2 + DFFD | SimCLR contrastive pretraining of the ResNet-50 backbone |
| 2 | `stage2_resnet_bilstm.yaml` | DFDC + FF++ + Celeb-DF | multi-task fine-tuning (focal + method + temporal + contrastive) |
| 3 | `stage3_robust.yaml` | + DeeperForensics + WildDeepfake | domain adaptation, PGD adversarial training, mixup, noise augmentation |
| KD | `effnet_distill.yaml` | stage-3 data | EfficientNet-B2 student for mobile |
| ens | `vit.yaml`, `video3d.yaml`, `ensemble.yaml` | stage-2 data | extra members + adaptive gated voting |

On the cluster (Altair PBS Pro, 1 GPU; set `ngpus=4` in the `#PBS -l select` line to use more):

```bash
bash jobs/run_pipeline.sh                      # submits every stage as dependent jobs
# or one at a time
qsub -v CONFIG=configs/experiments/stage2_resnet_bilstm.yaml jobs/pbs/train.pbs
qsub -v CONFIG=configs/experiments/stage2_resnet_bilstm.yaml,OVERRIDES="data.batch_size=6 train.grad_accum=2" jobs/pbs/train.pbs
tensorboard --logdir runs                      # set logging.wandb=true to also log to W&B
```

Training auto-resumes from `<out_dir>/last.pth`, so you can resubmit a job that hit its walltime. Any config value can be overridden as `key=value`. With `qsub -v`, separate overrides with spaces, not commas.

Locally or interactively:

```bash
torchrun --standalone --nproc_per_node=1 -m deepshield.cli train -c configs/experiments/stage2_resnet_bilstm.yaml
```

Memory guide (an estimate, not measured here, so check `nvidia-smi` on your first run): a batch of 8 clips × 16 frames × 224² with ResNet-50 and bf16 AMP should need roughly 15–22 GB. On a 16 GB GPU use `data.batch_size=4 train.grad_accum=2`, or `data.clip_len=8`.

## 4. Evaluation (PRD §8, deliverables D2–D4)

```bash
CK=runs/stage3_robust/best.pth; CFG=configs/experiments/eval_all.yaml
deepshield evaluate    -c $CFG --checkpoint $CK --out outputs/eval/benchmark    # per-dataset + generalisation gap, ROC, confusion
deepshield kyc-eval    -c $CFG --checkpoint $CK --out outputs/eval/kyc          # 360p, blur, JPEG, light, occlusion, combined, 80:20, latency
deepshield robust-eval -c $CFG --checkpoint $CK --out outputs/eval/robustness   # FGSM/PGD/AutoAttack, smoothing, detection, visualisations
deepshield report --dirs outputs/eval/* --out outputs/eval/REPORT.md            # tables + PRD target PASS/FAIL
# or everything at once:  qsub -v CKPT=$CK,TAG=stage3 jobs/pbs/evaluate.pbs
```

The decision threshold defaults to the **validation EER point**, which is stored in the checkpoint. Set `eval.threshold=0.5` to override it.

## 5. Deployment (PRD §6.4, D1/D5)

```bash
deepshield export -c configs/experiments/eval_all.yaml --checkpoint $CK     # ONNX + parity check, INT8 ONNX, torch dynamic INT8, TensorRT (if trtexec)
deepshield profile --checkpoint $CK                                          # per-stage latency (spatial / temporal / fusion / classify)
deepshield predict --checkpoint outputs/export/deepshield.onnx --inputs selfie.jpg liveness.mp4
```

```python
from deepshield.deploy import DeepShieldPredictor
p = DeepShieldPredictor("runs/stage3_robust/best.pth", robustness_check=True)
r = p.predict("liveness.mp4")   # prob_fake, decision (accept / reject / manual_review), frame scores, robustness confidence
```

## 6. Notes and deviations from the PRD

* **Contrastive loss.** The PRD writes `L = ||f(x_real) − f(x_fake)||²`. Minimising that literally would pull real and fake embeddings *together*, so a margin contrastive loss is used instead (same-class pairs pulled together, real–fake pairs pushed ≥ m apart).
* **AutoAttack on a 2-class model.** The targeted APGD-DLR and FAB components need ≥ 3 classes, so the untargeted APGD-CE + Square subset is run (official `autoattack` package if installed, otherwise the built-in implementations).
* **Certified robustness.** Randomized smoothing certifies L2 radii. An L∞ ε = 8/255 ball corresponds to an L2 radius of about 3.5 at 224², far beyond what smoothing can certify. The report therefore gives certified accuracy at L2 radii and empirical L∞ accuracy separately.
* **Focal α.** The class-balanced sampler already equalises classes, so the default is α=[0.5, 0.5]. Use α=[0.25, 0.75] with `data.balanced_sampling=false` (natural 80:20 sampling) to avoid correcting the imbalance twice.
* **Voice deepfakes and real-time streaming** are out of scope, as in PRD §3.2. Mega-MMDF is used for its video stream only.
* The accuracy targets (≥95% DFDC, etc.) are what the evaluation checks. Whether they are reached depends on the real training runs. The synthetic smoke data only verifies that the pipeline works.
