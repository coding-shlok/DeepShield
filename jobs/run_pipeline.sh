#!/usr/bin/env bash
# Submit the whole PRD training pipeline as dependent PBS jobs (each starts when the previous succeeds).
#   bash jobs/run_pipeline.sh            # all stages
#   SKIP_PRETRAIN=1 bash jobs/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
dep() { [ -n "${1:-}" ] && echo "-W depend=afterok:$1" || true; }

PRE=""
if [ -z "${SKIP_PRETRAIN:-}" ]; then
  PRE=$(qsub jobs/pbs/pretrain.pbs); echo "stage1 simclr   : $PRE"
  S2_OVR=""
else
  S2_OVR="model.backbone.checkpoint=null"
fi
S2=$(qsub $(dep "$PRE") -v CONFIG=configs/experiments/stage2_resnet_bilstm.yaml,OVERRIDES="$S2_OVR" jobs/pbs/train.pbs)
echo "stage2 train    : $S2"
S3=$(qsub $(dep "$S2") -v CONFIG=configs/experiments/stage3_robust.yaml jobs/pbs/train.pbs);         echo "stage3 robust   : $S3"
KD=$(qsub $(dep "$S3") -v CMD=distill,CONFIG=configs/experiments/effnet_distill.yaml jobs/pbs/train.pbs); echo "effnet distill  : $KD"
VIT=$(qsub $(dep "$S2") -v CONFIG=configs/experiments/vit.yaml,OVERRIDES="$S2_OVR" jobs/pbs/train.pbs); echo "vit member      : $VIT"
EV=$(qsub $(dep "$S3") -v CKPT=runs/stage3_robust/best.pth,TAG=stage3 jobs/pbs/evaluate.pbs);       echo "stage3 eval     : $EV"
EN=$(qsub -W depend=afterok:$KD:$VIT -v CMD=fit-ensemble,CONFIG=configs/experiments/ensemble.yaml jobs/pbs/train.pbs)
echo "ensemble        : $EN"
echo "monitor with: qstat -u $USER   |   tensorboard --logdir runs"
