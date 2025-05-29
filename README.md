## Installation

> **Prerequisites**  
> * Install miniconda or anaconda if you don't already have conda installed  
> https://www.anaconda.com/download/success

> * If you’re using Miniconda:
> Git isn’t bundled. Run this once, then continue:
>> ```bash
>> conda install git
>> ```

### 1 — Create & activate the environment

```bash
conda create -y -n clonalisa -c conda-forge python=3.10.12 r-base=4.3 pip
conda activate clonalisa
```

### 2 — Download and enter clonalisa repository

```bash
git clone https://github.com/timboGithubMan/clonalisa
cd clonalisa
```

### 3 — Install PyTorch and ClonaLisa

Pick **one** command that matches your hardware:

#### • CPU-only

```bash
pip install torch torchvision torchaudio -e . --index-url https://download.pytorch.org/whl/cpu
```

#### • CUDA 11.8 — most GPUs

```bash
pip install torch torchvision torchaudio -e . --index-url https://download.pytorch.org/whl/cu118
```

#### • CUDA 12.x — RTX 5000-series & newer

```bash
pip install torch torchvision torchaudio -e . --index-url https://download.pytorch.org/whl/cu128
```
