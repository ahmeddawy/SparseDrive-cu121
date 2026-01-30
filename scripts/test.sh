export CUBLAS_WORKSPACE_CONFIG=:4096:8

# bash ./tools/dist_test.sh \
#     /home/oem/Practice/sparsedrive_law/sparsedrive_cu126/projects/configs/sparsedrive_small_stage2_wm_optimized_v2.py \
#     /home/oem/Practice/sparsedrive_law/sparsedrive_cu126/work_dirs/sparsedrive_stage2_wm/iter_140650.pth \
#     1 \
#     --eval bbox
#     # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl



bash tools/dist_test.sh \
  projects/configs/sparsedrive_small_stage2_wm_optimized_v2.py \
  work_dirs/sparsedrive_stage2_wm/iter_140650.pth \
  1 \
  --eval bbox