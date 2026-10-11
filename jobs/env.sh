# Sourced by every PBS job. Edit for your cluster (module names, conda env, scratch path).
# module load cuda/12.1          # if your site uses environment modules
# module load anaconda3
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${DEEPSHIELD_ENV:-deepshield}"
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1
# Large temporary files (archive members being processed) – point at node-local scratch if you have it
export TMPDIR=${DEEPSHIELD_TMP:-${TMPDIR:-/tmp}}
# number of GPUs PBS gave us (falls back to all visible)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NGPUS=${NGPUS:-1}; [ "$NGPUS" -lt 1 ] && NGPUS=1
export NGPUS
