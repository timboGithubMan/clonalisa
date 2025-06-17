![ClonaLiSA](https://raw.githubusercontent.com/timboGithubMan/clonalisa/ebb9c1a510d5feccab17cfa6de91b3e2b310d534/logo.png)
# Installation

> **Prerequisites**  
> * Install miniconda or anaconda if you don't already have conda installed  
> https://www.anaconda.com/download/success

> * Highly recommended for a faster install
>> ```bash
>> conda config --set solver libmamba
>> ```

> * If you’re using Miniconda:
> Git isn’t bundled. Run this once, then continue:
>> ```bash
>> conda install git
>> ```

## Step 1 - Setup Environment

#### Option 1: GPU Install

>> ```bash
>> conda create -y -n clonalisa -c conda-forge python=3.10.12 r-base=4.3 r-ggplot2 r-dplyr r-stringr r-forcats r-tidyr r-tibble r-nlme r-emmeans r-broom.mixed r-gridextra r-codetools cupy=13.4 cuda-version=12.8 cuda-nvrtc=12.8
>> conda activate clonalisa
>> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
>> ```

#### Option 2: CPU-only
>> ```bash
>> conda create -y -n clonalisa -c conda-forge python=3.10.12 r-base=4.3 r-ggplot2 r-dplyr r-stringr r-forcats r-tidyr r-tibble r-nlme r-emmeans r-broom.mixed r-gridextra r-codetools
>> conda activate clonalisa
>> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
>> ```

## Step 2 — Install ClonaLiSA
```bash
git clone https://github.com/timboGithubMan/clonalisa
cd clonalisa
pip install -e .
```

## Step 3
watch your colonies self-organize into turquoise WGCNA cults
