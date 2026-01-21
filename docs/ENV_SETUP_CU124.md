# SparseDrive CU124 Environment Setup (Fresh Env)

This document captures the clean, reproducible steps used to build a
working CUDA 12.4 + PyTorch 2.4 environment for this repo. It keeps all
CUDA tooling inside the conda env (no system CUDA/NVCC required). The
OpenMMLab core stack is installed using OpenMIM; `mmcv` is built from
source (ninja) with explicit arch flags to ensure `mmcv._ext` is available
on CUDA 12.4. `flash-attn` and the repo's custom CUDA ops are built from
source using the same CUDA build env.

Tested on:
- Python 3.10
- PyTorch 2.4.0+cu124
- H100 (sm_90), RTX 6000 Pro/Ada (sm_89)

## 0) Prereqs

- NVIDIA driver that supports CUDA 12.4+
- Conda installed and working
- Repo path: `/home/oem/Practice/sparsedrive_law/sparsedrive_cu126`

## 1) Create and activate the env

```bash
conda create -n sparsedrive_cu124 python=3.10 -y
conda activate sparsedrive_cu124
```

## 2) Install PyTorch (CUDA 12.4 wheels)

```bash
python -m pip install --upgrade pip
python -m pip install \
  torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

## 3) CUDA toolchain + build env (required before any CUDA builds)

If `nvcc` is missing from `$CONDA_PREFIX/bin`, install the toolkit:

```bash
conda install -c nvidia cuda-toolkit=12.4 -y
```

Set arch flags for H100 (sm_90) and RTX 6000 Pro/Ada (sm_89). If you also
target RTX A6000 (Ampere, sm_86), include 8.6:

```bash
export TORCH_CUDA_ARCH_LIST="9.0;8.9;8.6"
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
```

## 4) Install OpenMMLab core stack (OpenMIM + ninja)

Install OpenMIM and ninja, then mmengine:

```bash
python -m pip install openmim ninja
mim install mmengine==0.10.4
```

Build `mmcv` from source (CUDA 12.4). This uses ninja automatically.
Note: `mmcv` does not support PEP 660 editable installs, so use a
non-editable install or build in-place.

```bash
python -m pip uninstall -y mmcv
git clone https://github.com/open-mmlab/mmcv.git -b v2.1.0
cd mmcv
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install -v . --no-build-isolation
cd -
```

Install mmdet and mmdet3d via OpenMIM. Use `--no-deps` so `mmcv` is not
overwritten:

```bash
mim install mmdet==3.3.0 --no-deps
mim install mmdet3d==1.4.0 --no-deps
```

Verify `mmcv._ext`:

```bash
python - <<'PY'
import mmcv, mmcv.ops
print("mmcv:", mmcv.__version__)
print("mmcv._ext ok")
PY
```

## 5) Install repo Python deps (pinned)

Lock NumPy < 2 to avoid ABI breakage in matplotlib/scipy/sklearn.
Install OpenCV without dependencies to prevent NumPy from being upgraded.

```bash
python -m pip install \
  numpy==1.26.4 pandas==2.2.2 scipy==1.11.4 \
  shapely==1.8.5 matplotlib==3.5.2 \
  prettytable==3.7.0 pyquaternion==0.9.9 \
  nuscenes-devkit==1.1.10 yapf==0.40.2 tensorboard==2.17.0 \
  motmetrics==1.4.0 scikit-learn==1.3.2 tqdm==4.66.4 \
  pillow==10.3.0 einops==0.7.0 pycocotools==2.0.7 \
  urllib3==1.26.18 psutil==5.9.8 terminaltables==3.1.10

# ensure no conda OpenCV is installed, then pin opencv-python
conda remove -y opencv py-opencv
python -m pip install --no-deps --force-reinstall opencv-python==4.8.1.78
```

## 6) Build flash-attn (source)

Use the same `TORCH_CUDA_ARCH_LIST` from step 3. Add `lib64` for the
linker if needed:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

## 7) Build custom CUDA ops (source)

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
cd /home/oem/Practice/sparsedrive_law/sparsedrive_cu126/projects/mmdet3d_plugin/ops
python -m pip install . --no-build-isolation
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
- Warnings about missing keys when loading `ckpt/sparsedrive_stage1.pth`
  into stage2 are expected (motion/planning heads are absent in stage1).
