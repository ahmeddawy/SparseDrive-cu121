# SparseDrive CU121 Environment Setup (Fresh Env)

This document captures the exact steps used to reproduce the working
CUDA 12.1 + PyTorch 2.4 environment for this repo. It avoids touching
system CUDA/NVCC and uses a Conda-provided toolchain inside the env.

Tested on:
- Python 3.10
- PyTorch 2.4.0+cu121
- RTX 4090 Laptop GPU (sm_89), RTX 6000 Ada (sm_89), H100 (sm_90)

## 0) Prereqs

- NVIDIA driver that supports CUDA 12.1+
- Conda installed and working
- Repo path: `/home/oem/Practice/sparsedrive_law/sparsedrive_cu126`

## 1) Create and activate the env

```bash
conda create -n sparsedrive_cu121 python=3.10 -y
conda activate sparsedrive_cu121
```

## 2) Install PyTorch (CUDA 12.1 wheels)

```bash
python -m pip install --upgrade pip
python -m pip install \
  torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

## 3) Install OpenMMLab core stack

```bash
python -m pip install mmengine==0.10.4
python -m pip install mmcv==2.1.0
python -m pip install mmdet==3.3.0
python -m pip install mmdet3d==1.4.0
```

## 4) Install repo Python deps

NuScenes devkit 1.1.10 requires matplotlib<=3.5.2 and shapely<=1.8.5.

```bash
python -m pip install \
  numpy==1.26.4 pandas==2.2.2 scipy==1.11.4 \
  shapely==1.8.5 matplotlib==3.5.2 \
  opencv-python==4.8.1.78 prettytable==3.7.0 pyquaternion==0.9.9 \
  nuscenes-devkit==1.1.10 yapf==0.40.2 tensorboard==2.17.0 \
  motmetrics==1.4.0 scikit-learn==1.3.2 tqdm==4.66.4 \
  pillow==10.3.0 einops==0.7.0 pycocotools==2.0.7 \
  urllib3==1.26.18 psutil==5.9.8
```

## 5) CUDA toolchain inside the env (no system CUDA usage)

If `nvcc` is not inside the env, install a CUDA toolkit package:

```bash
conda install -c nvidia cuda-toolkit=12.1 -y
```

Then export CUDA paths to prefer the env’s compiler + headers:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
```

## 6) Build flash-attn

Set arch list for H100 (sm_90) and RTX 6000 Ada (sm_89):

```bash
export TORCH_CUDA_ARCH_LIST="9.0;8.9"
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

## 7) Build custom CUDA ops

```bash
cd /home/oem/Practice/sparsedrive_law/sparsedrive_cu126/projects/mmdet3d_plugin/ops
python -m pip install -e . --no-build-isolation
cd -
```

## 8) Sanity checks

```bash
python - <<'PY'
import torch, mmcv, mmdet, mmengine, mmdet3d, flash_attn
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("mmcv:", mmcv.__version__)
print("mmengine:", mmengine.__version__)
print("mmdet:", mmdet.__version__)
print("mmdet3d:", mmdet3d.__version__)
print("flash-attn:", flash_attn.__version__)
PY
```

## 9) Data prep

Place NuScenes at `./data/nuscenes` (v1.0-trainval). Then:

```bash
cd /home/oem/Practice/sparsedrive_law/sparsedrive_cu126
sh scripts/create_data.sh
sh scripts/kmeans.sh
```

## 10) Test and train

```bash
sh scripts/test.sh
sh scripts/train.sh
```

Notes:
- If you only have a partial NuScenes dataset, evaluation in this repo
  filters GTs to match prediction tokens so detection/tracking/map
  evaluation can proceed.
- The warnings about missing keys when loading `ckpt/sparsedrive_stage1.pth`
  into stage2 are expected (motion/planning heads are absent in stage1).

## 11) Optional reproducibility (deterministic runs)

If you enable deterministic mode, set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 12) Moving to CUDA 12.6 later

When cu126 wheels are available, only change:
- PyTorch wheel index/version
- `cuda-toolkit` version in the env
- `flash-attn` build uses the same `TORCH_CUDA_ARCH_LIST`
