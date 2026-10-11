#!/usr/bin/env bash
# One-time environment setup on the cluster login node.
#   bash jobs/setup_env.sh            (CUDA 12.1 wheels by default; CUDA=cu118 bash jobs/setup_env.sh)
set -euo pipefail
CUDA=${CUDA:-cu121}
ENV=${DEEPSHIELD_ENV:-deepshield}
conda create -y -n "$ENV" python=3.10
eval "$(conda shell.bash hook)"
conda activate "$ENV"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA}"
pip install -r requirements.txt
pip install -e .
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
deepshield datasets info
