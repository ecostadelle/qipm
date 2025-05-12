# Quality-weighted Intervention in Prediction Measure (QIPM)

This repository contains the implementation of the experiments described in the paper:

> **A New Approach to Handle Data Shift Based on Feature Importance Measurement**  
> Ewerton Luiz Costadelle, Marcelo Rodrigues de Holanda Maia, Alexandre Plastino, Alex Alves Freitas  

## Overview

QIPM is a novel feature importance metric for Random Forest models, incorporating quality metrics such as accuracy and F1-score. This repository contains the code to reproduce the experiments presented in the paper using the TableShift benchmark.

---

## Installation

It is recommended to use **Python 3.9** to ensure compatibility.

### Dependencies

For example, in Ubuntu distribution you can install the Python and the project dependencies by running the following command:

1. Add the deadsnakes PPA (which contains Python 3.9)
```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
```

2. Install Python 3.9 and dependencies
```bash
sudo apt install -y python3.9 python3.9-venv python3.9-dev python3.9-distutils, build-essential
```

3. Create a virtual environment
```bash
python3.9 -m venv .venv
```

4. Activate the virtual environment
```bash
source .venv/bin/activate
```

5. Install the package dependencies
```bash
pip install -r requirements.txt
```

6. Compile and install the package
```bash
pip install src/
```

---

## Downloading the Datasets

The datasets used in the experiments can be downloaded using the script `cache_tableshift.py`, available in the root directory of the repository. Run the following command:

```bash
python cache_tableshift.py
```

This script will download and store the necessary datasets for the experiments. Some datasets require credentialed download. See [Gardner _et al._ (2023)](https://arxiv.org/abs/2312.07577) for more information.

**Note:** Some datasets are large and may take significant time to download. Ensure you have a stable internet connection and enough disk space before proceeding.

---

## Running the Experiments

After installing dependencies and downloading the datasets, you can reproduce the experiments by running:

```bash
python script.py
```

---

## Citation

If you use this code in your work, please cite the corresponding paper:

```
@article{costadelle2025qipm,
  author    = {Ewerton Luiz Costadelle and Marcelo Rodrigues de Holanda Maia and Alexandre Plastino and Alex Alves Freitas},
  title     = {A New Approach to Handle Data Shift Based on Feature Importance Measurement},
  journal   = {Under submission},
  year      = {2025}
}
```

---

## Contact

For questions or suggestions, please contact ecostadelle@id.uff.br.

---