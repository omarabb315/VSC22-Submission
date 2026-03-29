#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

source vsc22_env/bin/activate

# Avoid OMP crash on macOS with multiple linked OpenMP libraries
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

VCDB_DIR="${1:-vcdb_core}"
MAX_VIDEOS="${2:-}"

echo "=== VSC22 Video Copy Detection Pipeline ==="
echo "VCDB dir:    ${VCDB_DIR}"
echo "Max videos:  ${MAX_VIDEOS:-all}"
echo ""

# --- Step 1: Download checkpoints if missing ---
CKPT_DIR="$REPO_ROOT/VSC22-Descriptor-Track-1st/checkpoints"
if [ ! -f "$CKPT_DIR/vit_v68.torchscript.pt" ]; then
    echo "=== Downloading pretrained checkpoints ==="
    mkdir -p "$CKPT_DIR"
    TAR="$CKPT_DIR/../checkpoints.tar.gz"
    gdown 1GL0xhTTSHav_iG79yJ1jqgQcmuJFs_lF -O "$TAR"
    tar -xzf "$TAR" -C "$CKPT_DIR/"
    rm -f "$TAR"
    # Flatten one directory level if archive wrapped files in a subfolder
    if [ ! -f "$CKPT_DIR/vit_v68.torchscript.pt" ]; then
        for sub in "$CKPT_DIR"/*; do
            if [ -d "$sub" ] && [ -f "$sub/vit_v68.torchscript.pt" ]; then
                mv "$sub"/* "$CKPT_DIR/" 2>/dev/null || true
                rmdir "$sub" 2>/dev/null || true
                break
            fi
        done
    fi
fi

# --- Step 2: Download VCDB if missing ---
if [ ! -d "$VCDB_DIR" ] || [ -z "$(ls -A "$VCDB_DIR" 2>/dev/null)" ]; then
    echo "=== Downloading VCDB core dataset ==="
    gdown --folder 0B-b0CY525pH8NjdxbFNGY0JJdGs -O "$VCDB_DIR"
fi

# --- Step 3: Prepare data ---
echo "=== Preparing data ==="
PREP_CMD="python prepare_data.py --vcdb_dir $VCDB_DIR"
if [ -n "$MAX_VIDEOS" ]; then
    PREP_CMD="$PREP_CMD --max_videos $MAX_VIDEOS"
fi
$PREP_CMD

# --- Step 4: Descriptor Track - reference features ---
echo "=== Descriptor Track: extracting reference features ==="
cd "$REPO_ROOT/VSC22-Descriptor-Track-1st/infer"
export PYTHONPATH=$PYTHONPATH:./
bash infer_ref.sh

# Move train_refs.npz to data/
mkdir -p data
cp outputs/train_refs.npz data/ 2>/dev/null || true

# --- Step 5: Descriptor Track - query features ---
echo "=== Descriptor Track: extracting query features ==="
python3 extract_query_feats.py --split test

# --- Step 6: Descriptor Track - retrieval ---
echo "=== Descriptor Track: retrieval ==="
bash eval.sh

# --- Step 7: Matching Track ---
echo "=== Matching Track: segment-level detection ==="
cd "$REPO_ROOT/VSC22-Matching-Track-1st/infer"
export PYTHONPATH=$PYTHONPATH:./
python3 infer_matching.py --split test

echo ""
echo "=== Pipeline Complete ==="
echo "Descriptor outputs: $REPO_ROOT/VSC22-Descriptor-Track-1st/infer/outputs/"
echo "Matching results:   $REPO_ROOT/VSC22-Descriptor-Track-1st/infer/outputs/matching/test_matching.csv"
echo "ID mapping:         $REPO_ROOT/VSC22-Descriptor-Track-1st/id_mapping.json"
