export CUBLAS_WORKSPACE_CONFIG=:4096:8

bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    ckpt/sparsedrive_stage2.pth \
    1 \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl
