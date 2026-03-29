projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir
export KMP_DUPLICATE_LIB_OK=TRUE
if [[ "$(uname -s)" == "Darwin" ]]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
else
    _c=$(nproc 2>/dev/null || echo 8)
    [ "${_c}" -gt 16 ] && _c=16
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${_c}}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
fi

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
