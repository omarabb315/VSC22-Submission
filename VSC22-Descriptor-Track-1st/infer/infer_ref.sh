#!/bin/bash
projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

index=(0 1 2 3)
models=("swinv2_v115" "swinv2_v107" "swinv2_v106" "vit_v68")
img_sizes=(256 256 256 384)

jpg_zips_path="../data/jpg_zips"

# Detect GPU count (0 on macOS / CPU-only systems)
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
else
    gpu_count=0
fi
echo "Detected ${gpu_count} GPU(s)"

for i in ${index[@]}
do
if [[ ${gpu_count} -gt 0 ]]; then
    # --- Multi-GPU distributed mode ---
    # train refs
    torchrun --nproc_per_node=${gpu_count} extract_ref_feats.py \
            --zip_prefix ${jpg_zips_path}  \
            --input_file "train/train_ref_vids.txt" \
            --save_file train_refs \
            --save_file_root ./outputs/${models[i]} \
            --batch_size 2 \
            --input_file_root "../data/meta/" \
            --dataset "vsc" \
            --checkpoint_path ../checkpoints/${models[i]}.torchscript.pt \
            --transform "vit" \
            --img_size ${img_sizes[i]}

    # test refs
    torchrun --nproc_per_node=${gpu_count} extract_ref_feats.py \
            --zip_prefix ${jpg_zips_path}  \
            --input_file "test/test_ref_vids.txt" \
            --save_file test_refs \
            --save_file_root ./outputs/${models[i]} \
            --batch_size 2 \
            --input_file_root "../data/meta/" \
            --dataset "vsc" \
            --checkpoint_path ../checkpoints/${models[i]}.torchscript.pt \
            --transform "vit" \
            --img_size ${img_sizes[i]}
else
    # --- Single-process CPU/MPS mode (macOS or no GPU) ---
    # train refs
    python3 extract_ref_feats.py \
            --zip_prefix ${jpg_zips_path}  \
            --input_file "train/train_ref_vids.txt" \
            --save_file train_refs \
            --save_file_root ./outputs/${models[i]} \
            --batch_size 1 \
            --input_file_root "../data/meta/" \
            --dataset "vsc" \
            --checkpoint_path ../checkpoints/${models[i]}.torchscript.pt \
            --transform "vit" \
            --img_size ${img_sizes[i]}

    # test refs
    python3 extract_ref_feats.py \
            --zip_prefix ${jpg_zips_path}  \
            --input_file "test/test_ref_vids.txt" \
            --save_file test_refs \
            --save_file_root ./outputs/${models[i]} \
            --batch_size 1 \
            --input_file_root "../data/meta/" \
            --dataset "vsc" \
            --checkpoint_path ../checkpoints/${models[i]}.torchscript.pt \
            --transform "vit" \
            --img_size ${img_sizes[i]}
fi
done

# concat and reduce dim, finally sn
python3 concat_pca_sn.py
