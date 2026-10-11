# Data pipeline

## Processed format

Every dataset ends up in the same layout, which is the only thing training reads:

```
data/processed/<dataset>/
  manifest.csv          sample_id, dataset, label (0 real / 1 fake), method, group, path, n_frames, split
  faces/<sample_id>/000.jpg … 031.jpg     (256×256 face crops, margin 1.3, temporally smoothed boxes)
```

* `group` is the identity or source video. Splits are assigned **per group**, so an FF++ original and all of its manipulations, or a DFDC original and its fakes, always land in the same split. This prevents identity leakage.
* Official splits are used where they exist: FF++ JSON splits, the Celeb-DF test list, DeeperForensics lists, and the WildDeepfake train/test archives. When a dataset only has an official test split, validation is carved out of train. Everything else gets a stratified 70/15/15 split (seed 42).
* `method` drives the auxiliary multi-task head (`Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `FaceShifter`, `gan_synthesis`, `other_fake`).

## Location syntax (`source.locations` in `configs/data/*.yaml`)

| Form | Behaviour |
|---|---|
| `/path/dir` | walked recursively |
| `/path/file.zip`, `.tar(.gz)` | members read directly from the archive (no extraction) |
| `https://…/x.zip` | **HTTP Range reads**: no local copy (falls back to temp download if Range is unsupported) |
| `https://…/x.tar.gz` | streamed sequentially: no local copy |
| `gdrive://<file_id>` | gdown to temp, deleted after processing |
| `kaggle://<competition>/<file>` | Kaggle CLI to temp (needs `~/.kaggle/kaggle.json`), deleted after processing |

`source.tmp_dir` (or `$DEEPSHIELD_TMP`) picks where temporary files go. Point it at node-local scratch.

## Per-dataset notes

| Dataset | Labels / groups | Tips |
|---|---|---|
| FaceForensics++ | `original_sequences/youtube` = real, `manipulated_sequences/<method>` = fake. Group = official pair. | Set `download_script` to the official script. c23 is about 10 GB for everything. |
| DFDC | `metadata.json` per part (`label`, `original`). Group = original video. | Start with `train_sample_videos.zip` via Kaggle, then add the `dfdc_train_part_N.zip` URLs. |
| Celeb-DF v2 | `Celeb-synthesis` = fake, `Celeb-real` / `YouTube-real` = real. Official test list. | Group = celebrity id. |
| DeeperForensics-1.0 | `manipulated_videos/<variant>` = fake, `source_videos` = real. | Perturbed variants (`end_to_end_level_*`) are exactly the PRD's mobile-noise robustness data. |
| WildDeepfake | Archive name gives the label and split (`fake_train.tar.gz`). | Pre-cropped frames, so `detector: none`. |
| UADFV | `real/`, `fake/` folders | Unseen-dataset check (PRD §9). |
| VGGFace2 | real identities, group = `n0xxxxx` | `max_per_identity: 50` keeps SimCLR data at about 450k faces. |
| DFFD | ffhq/celeba = real, stylegan/pggan/stargan/faceapp = `gan_synthesis` | Stage-1 synthetic familiarity. |
| DeepfakeBench | imports its `dataset_json` + preprocessed frames | Quickest way to get UADFV and others already cropped. |
| DeePhy, Mega-MMDF, anything else | generic `folder` adapter with regex `rules` | Mega-MMDF: video stream only. |

## Face detection

`detector: retinaface` uses insightface's RetinaFace (`buffalo_l` detection model) on the GPU. It falls back to MTCNN (facenet-pytorch), then to full-frame crops with a warning. Per video: sample `n_frames` uniformly, pick the largest confident face, EMA-smooth the box (reduces crop jitter that could look like a temporal artifact), expand by 1.3, crop square, resize to 256.

## KYC capture simulation (PRD §5.3)

`deepshield/data/augment.py::KYCPerturbation`. Parameters are sampled once per clip so they stay temporally consistent. The exception is hand-tremor jitter, which is per frame.

| Perturbation | Training range | Eval preset |
|---|---|---|
| resolution 1080p→360p | scale 0.25–0.6 | `res_360p` (1/3) |
| motion blur + tremor jitter | 3–5 px, ±5 px | `motion_blur_5px` |
| JPEG (WhatsApp/Telegram) | Q 60–80 | `jpeg_q70`, `jpeg_q60` |
| brightness | ±30 % | `brightness_minus30/plus30` |
| shadow casting | strength 0.2–0.5 | `shadow` |
| occlusion: mask, glasses, beard shadow, patch | random | `occlusion_mask`, `occlusion_glasses` |
| sensor noise | σ 2–8 | `sensor_noise` |
| **combined** | — | 360p + 5 px blur + JPEG 70 + −30 % light |

Class imbalance (80:20): training uses a class-balanced sampler and focal loss. Evaluation re-samples the test set to 80:20 (20 bootstrap repeats) and reports macro-F1, APCER and BPCER.
