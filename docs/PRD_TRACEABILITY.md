# PRD → implementation map

| PRD item | Implementation |
|---|---|
| §4.1 Input module ([-1,1], keyframes) | `data/extract.py`, `data/datasets.py`, `models/detector.py::InputNormalize` |
| §4.1 Feature extractor (ResNet-50, EfficientNet, 2048-d, multi-scale) | `models/backbones.py::SpatialBackbone` (timm, feature pyramid, SRM noise stream) |
| §4.1 Temporal module (BiLSTM / 3D-CNN) | `models/temporal.py::BiLSTMTemporal`, `models/video3d.py::Video3DNet` |
| §4.1 Ensemble classifier (multi-head attention fusion) | `models/fusion.py::AttentionFusion` (4 heads), `models/ensemble.py::AdaptiveEnsemble` |
| §4.1 Robustness filter | `robustness/detection.py` (feature squeezing, robustness confidence) |
| §4.2 Certified robustness, randomized smoothing | `robustness/smoothing.py::Smooth` (Cohen et al.) |
| §5.1 Multi-stage datasets (VGGFace2, DFFD, DFDC, FF++, Celeb-DF, DeeperForensics, WildDeepfake, DeepfakeBench, DeePhy, Mega-MMDF, UADFV) | `data/sources/registry.py`, `configs/data/*.yaml` |
| §5.3 KYC augmentation (360p, blur ±5 px, JPEG 60–80, ±30 % light, occlusion, 80:20) | `data/augment.py`, `data/datasets.py::sample_weights/subsample_ratio` |
| §6.2 Phase 1 SimCLR on VGGFace2 | `training/pretrain.py`, `configs/experiments/stage1_simclr.yaml` |
| §6.2 Phase 2 multi-task fine-tuning | `losses.py::MultiTaskLoss`, `stage2_resnet_bilstm.yaml` |
| §6.2 Phase 3 adversarial training, mixup, DeeperForensics/WildDeepfake | `training/trainer.py`, `stage3_robust.yaml` |
| §6.3 Focal / contrastive / temporal / adversarial losses; AdamW + warm-up + clipping | `losses.py`, `training/engine.py` |
| §6.4 Latency budget per stage | `deploy/profile.py::profile_model` |
| §6.4 INT8, distillation, batch processing, ONNX/TensorRT | `deploy/export.py`, `training/distill.py`, `DeepShieldNet.extract_spatial` (one vectorised pass) |
| §7 Train/val/test 70/15/15 | `data/manifest.py::assign_splits` (group-aware, stratified) |
| §8.1 Accuracy, AUC-ROC, EER | `evaluation/metrics.py` |
| §8.2 Generalisation gap, adversarial ε=8/255, latency, model size | `evaluation/evaluate.py`, `deploy/profile.py` |
| §8.3 Mobile-noise robustness, 80:20 balanced F1, false-rejection rate | `evaluation/evaluate.py::kyc_suite` (BPCER = real rejected) |
| §9 Fall back to reduced subset; 8-frame clips | `limit=` in prepare, `data.clip_len=8` |
| D1 checkpoints, INT8, ONNX, fixed seed | `runs/*/best.pth`, `deploy/export.py`, `utils/seed.py` |
| D2 benchmark report, ROC, confusion | `deepshield evaluate`, `deepshield report` |
| D3 KYC suite + latency | `deepshield kyc-eval` |
| D4 FGSM / PGD / AutoAttack / smoothing / visualisation | `deepshield robust-eval` |
| D5 inference pipeline, batch support | `deploy/inference.py::DeepShieldPredictor` |
| D6 repository, README, data docs, reproducibility guide | this repo, `docs/` |
| D7 thesis | out of scope for this build (as agreed) |
