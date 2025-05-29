# How to Install
## 1 – Create & enter the env (Python 3.10, R 4.3)
conda create -y -n clonalisa -c conda-forge python=3.10.12 r-base=4.3 pip  
conda activate clonalisa

## 2 – Grab the repo
git clone https://github.com/timboGithubMan/clonalisa  
cd clonalisa

## 3 – Install PyTorch and clonalisa  (choose one command that matches your setup)
### CPU-only
pip install torch torchvision torchaudio -e . --index-url https://download.pytorch.org/whl/cpu
### CUDA 11.8  → most GPUs
pip install torch torchvision torchaudio -e . --index-url https://download.pytorch.org/whl/cu118
### CUDA 12.x → RTX 5000-series & newer
pip install torch torchvision torchaudio -e . --index-url https://download.pytorch.org/whl/cu128
