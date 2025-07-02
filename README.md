# Advancing ADMET Prediction: A Hybrid Machine Learning Framework (Partex ADMETrix)

## Overview
This project implements a comprehensive machine learning framework for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties of chemical compounds. The framework utilizes a combination of molecular fingerprints (Morgan, Avalon, ErG) and RDKit physicochemical descriptors for molecular representation. It employs a selection of robust machine learning algorithms (LightGBM, XGBoost, CatBoost, and Feedforward Neural Networks) and leverages Optuna for hyperparameter optimization to achieve competitive performance across various ADMET benchmarks from the Therapeutics Data Commons (TDC).

## Author Information
- **Author**: Rohit Singh Yadav
- **Organization**: Partex NV

## Features
- **Hybrid Model Approach**: Utilizes LightGBM, XGBoost, CatBoost, and Feedforward Neural Networks (FFN) based on benchmark suitability.
- **Comprehensive Molecular Representation**:
    - Morgan Fingerprints
    - Avalon Fingerprints
    - ErG Fingerprints
    - ~200 RDKit Physicochemical Descriptors
- **Hyperparameter Optimization**: Employs Optuna with TPESampler for robust tuning.
- **Support for Diverse Tasks**: Handles both binary classification and regression ADMET endpoints.
- **Standardized Benchmarking**: Evaluated on 21 ADMET benchmarks from the Therapeutics Data Commons (TDC).
- **Robust Evaluation**: Results are averaged over 5 random seeds for reliable performance metrics.
- **Extensive Logging**: Detailed logging of the training process, hyperparameter optimization, and evaluation results.
- **GPU Acceleration**: Utilizes GPU for XGBoost, CatBoost, and FFN training where available.

## Dependencies
- Python 3.7+
- PyTorch (for FFN)
- RDKit-pypi
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Optuna
- PyTDC (Therapeutics Data Commons)
- tqdm (for progress bars)

## Installation

### 1. Clone the Repository
```bash
git clone the main repo
```
### 2. Setup Environment

Using Conda (Recommended):
It's recommended to create a Conda environment. You can create an environment.yml file with the following content, Then create and activate the environment:
```bash
conda env create -f environment.yml
conda activate admet_hybrid_env
```
Using Pip:
Alternatively, you can install packages using pip. Ensure you have Python 3.7+ and pip installed.
```bash
pip install torch torchvision torchaudio
pip install rdkit-pypi numpy pandas scikit-learn lightgbm xgboost catboost optuna PyTDC tqdm
```
(Note: For GPU support with PyTorch, XGBoost, and CatBoost, ensure you have the appropriate CUDA toolkit installed and install GPU-enabled versions of these libraries. Refer to their official documentation for specific instructions.)

### 3. Supported ADMET Properties

The framework is designed to work with the 22 ADMET benchmarks provided by the Therapeutics Data Commons, covering a wide range of endpoints.

Usage

The main script (e.g., admet_prediction.py - please adjust to your actual script name) orchestrates the entire benchmarking process.

```bash
python admet_prediction.py
```
```bash
Project Structure (Illustrative)
.
├── fingerprint_gen.py          # Utility script 
├── admet_prediction.py         # Main script to run TDC benchmarking
├── data/                       # Directory for TDC downloaded datasets (created by PyTDC)
├── model_output_v2.log         # Detailed logging file
├── README.md                   # Readme file
└── environment.yml             # Conda environment file
```
model_output_v2.log: Contains detailed logs of the entire process, including hyperparameter optimization trials, individual seed results, and final aggregated benchmark scores.

### 4. Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

### 5. License
Distributed under the License. See LICENSE.txt for more information.

### 6. Acknowledgements

1. Therapeutics Data Commons (TDC) for providing standardized datasets and benchmarks.

2. The RDKit community for the cheminformatics toolkit.

3. Developers of PyTorch, LightGBM, XGBoost, CatBoost, and Optuna.


## Powered By

<div align="center">
  <a href="https://tdcommons.ai/" target="_blank" rel="noopener noreferrer" style="display: inline-block; margin: 10px 15px;">
    <img src="https://tdcommons.ai/tdc_horizontal.png" alt="Therapeutics Data Commons (TDC)" height="55"/>
  </a>
  <a href="https://optuna.org/" target="_blank" rel="noopener noreferrer" style="display: inline-block; margin: 10px 15px;">
    <img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" alt="Optuna" height="60"/>
  </a>
</div>

<div align="center">
  <a href="https://pytorch.org/" target="_blank" rel="noopener noreferrer" style="display: inline-block; margin: 10px 15px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" alt="PyTorch" height="65"/>
  </a>
  <a href="https://www.rdkit.org/" target="_blank" rel="noopener noreferrer" style="display: inline-block; margin: 10px 15px;">
    <img src="https://gdb.unibe.ch/content/images/2021/10/rdkit.png" alt="RDKit" height="75"/>
  </a>
</div>

<div align="center">
  <a href="https://lightgbm.readthedocs.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block; margin: 10px 15px;">
    <img src="https://lightgbm.readthedocs.io/en/stable/_images/LightGBM_logo_black_text.svg" alt="LightGBM" height="50"/>
  </a>
  <a href="https://xgboost.ai/" target="_blank" rel="noopener noreferrer" style="display: inline-block; margin: 10px 15px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/XGBoost_logo.svg/800px-XGBoost_logo.svg.png" alt="XGBoost" height="50"/>
  </a>
  <a href="https://catboost.ai/" target="_blank" rel="noopener noreferrer" style="display: inline-block; margin: 10px 15px;">
    <img src="https://www.zdnet.com/a/img/resize/26f2b28389f2759d17260948a657511144c5b988/2017/07/18/d3f47c3e-8529-4855-a0e1-c686ee3b4007/orig.png?auto=webp&fit=crop&height=675&width=1200" alt="CatBoost" height="75"/>
  </a>
</div>
