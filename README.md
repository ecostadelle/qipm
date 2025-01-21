# Quality-weighted Intervention in Prediction Measure (QIPM)

This repository contains the implementation of the experiments described in the paper:

> **Quality-weighted Intervention in Prediction Measure: a novel approach that considers relevant aspects when coping with dataset shift problem**  
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

3. create a virtual env
```bash
python3.9 -m venv .venv
```

4. activate venv
```bash
source .venv/bin/activate
```

5. Compile and install this package and the dependecies
```bash
pip install .
```

---

## Downloading the Datasets

The datasets used in the experiments can be downloaded using the script `cache_tableshift.py`, available in the root directory of the repository. Run the following command:

```bash
python cache_tableshift.py
```

This script will download and store the necessary datasets for the experiments. Some datasets require credentialized download. See [Gardner _et al._ (2023)](https://arxiv.org/abs/2312.07577) for more infomation.

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
  title     = {Quality-weighted Intervention in Prediction Measure: a novel approach that considers relevant aspects when coping with dataset shift problem},
  journal   = {Under submission},
  year      = {2025}
}
```

---

## Contact

For questions or suggestions, please contact ecostadelle@id.uff.br.

---