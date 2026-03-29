#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== VSC22 Environment Setup ==="

python3 -m venv vsc22_env
source vsc22_env/bin/activate
pip install --upgrade pip setuptools wheel

# --- Detect platform and install PyTorch ---
if [[ "$(uname)" == "Darwin" ]]; then
    echo "=== macOS detected: installing PyTorch (CPU/MPS) ==="
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
    pip install faiss-cpu
else
    echo "=== Linux detected: installing PyTorch + CUDA 12.1 ==="
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install faiss-gpu-cu12 || { echo "WARN: faiss-gpu-cu12 failed, falling back to faiss-cpu"; pip install faiss-cpu; }
fi

# --- Numerical / scientific ---
pip install scipy scikit-learn pandas Pillow

# --- Repo dependencies ---
pip install "transformers>=4.36.0"
pip install lmdb ftfy
pip install "albumentations>=1.3.0"
pip install "lightning>=2.0.0"
pip install "pytorch-metric-learning>=2.0.0"
pip install "torchmetrics>=0.11.0"
pip install "tslearn>=0.5.2"
pip install yacs
pip install "timm>=0.9.12"
pip install loguru matplotlib
pip install tensorboard tensorboardx
pip install yapf
pip install git+https://github.com/openai/CLIP.git
pip install tqdm

# --- mmcv (training only, graceful failure OK for inference) ---
pip install -U openmim && mim install mmcv==2.1.0 \
    || echo "WARN: mmcv install failed (not needed for inference-only mode)"

# --- Utilities ---
pip install gdown ipykernel

# --- Register Jupyter kernel ---
python -m ipykernel install --user --name vsc22 --display-name "VSC22 (PyTorch 2.4)"

# --- Set up Matching Track symlinks ---
echo "=== Setting up Matching Track symlinks ==="
cd "$REPO_ROOT/VSC22-Matching-Track-1st"
ln -sf ../VSC22-Descriptor-Track-1st/checkpoints checkpoints 2>/dev/null || true
ln -sf ../VSC22-Descriptor-Track-1st/data data 2>/dev/null || true
mkdir -p infer
ln -sf ../../VSC22-Descriptor-Track-1st/infer/outputs infer/outputs 2>/dev/null || true
cd "$REPO_ROOT"

# --- Create output directories ---
mkdir -p VSC22-Descriptor-Track-1st/checkpoints
mkdir -p VSC22-Descriptor-Track-1st/infer/outputs
mkdir -p VSC22-Descriptor-Track-1st/data/videos/test/query
mkdir -p VSC22-Descriptor-Track-1st/data/videos/test/reference
mkdir -p VSC22-Descriptor-Track-1st/data/meta/test
mkdir -p VSC22-Descriptor-Track-1st/data/meta/train
mkdir -p VSC22-Descriptor-Track-1st/data/jpg_zips
mkdir -p VSC22-Matching-Track-1st/infer/outputs/matching

# --- Verify installation ---
echo ""
echo "=== Verifying installation ==="
python -c "
import torch, torchvision, numpy, pandas, sklearn, timm, PIL
print(f'PyTorch:      {torch.__version__}')
print(f'Torchvision:  {torchvision.__version__}')
print(f'NumPy:        {numpy.__version__}')
print(f'Pandas:       {pandas.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'timm:         {timm.__version__}')
print(f'Pillow:       {PIL.__version__}')
print(f'CUDA:         {torch.version.cuda or \"N/A\"}')
has_cuda = torch.cuda.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if has_cuda:
    print(f'Device:       CUDA - {torch.cuda.get_device_name(0)}')
elif has_mps:
    print(f'Device:       MPS (Apple Silicon)')
else:
    print(f'Device:       CPU only')
try:
    import faiss
    print(f'FAISS:        {faiss.__version__ if hasattr(faiss, \"__version__\") else \"installed\"}')
except ImportError:
    print('FAISS:        NOT INSTALLED')
"

echo ""
echo "=== Setup complete ==="
echo "Select 'VSC22 (PyTorch 2.4)' kernel in JupyterLab/Jupyter."
echo "Next: run 'source vsc22_env/bin/activate' to activate the environment."
echo ""
echo "OpenMP: run_pipeline.sh / infer_ref.sh / eval.sh set OMP_NUM_THREADS=1 on macOS"
echo "        and min(16, nproc) on Linux (override with export OMP_NUM_THREADS=N)."
