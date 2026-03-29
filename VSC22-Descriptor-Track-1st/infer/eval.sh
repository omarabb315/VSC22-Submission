projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# gt_path="../data/meta/val/eval_matches.csv"
# python3 -m vsc.baseline.sscd_baseline \
#     --query_features outputs/val_query_sn.npz \
#     --ref_features outputs/train_refs_sn.npz  \
#     --output_path ./outputs/ \
#     --overwrite \
#     --ground_truth ${gt_path} \

python3 -m vsc.baseline.sscd_baseline \
    --query_features outputs/test_query_sn.npz \
    --ref_features outputs/test_refs_sn.npz  \
    --output_path ./outputs/ \
    --overwrite \
