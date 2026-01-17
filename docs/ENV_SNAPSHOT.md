# Environment Snapshot (Conda + Pip)

This repo includes a frozen snapshot of the working `sparsedrive_cu121`
environment. Use it to recreate the same package set on a fresh machine.

## 1) Create the env from the snapshot

```bash
conda env create -f env/conda_env.yml
conda activate sparsedrive_cu121
```

Notes:
- The YAML was exported from a Linux system using conda-forge packages.
- You still need a compatible NVIDIA driver on the host.

## 2) Optional verification (pip snapshot)

```bash
python -m pip install -r env/pip_freeze.txt
```

This is optional and mainly for auditing. If a pip package cannot be
resolved on your system, prefer the conda YAML as the source of truth.

## 3) CUDA toolkit

If `nvcc` is missing inside the env, install it:

```bash
conda install -c nvidia cuda-toolkit=12.1 -y
```

Then export CUDA paths before building `flash-attn` or custom ops:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
```
