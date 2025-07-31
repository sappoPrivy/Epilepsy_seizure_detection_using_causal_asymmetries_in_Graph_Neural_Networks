# Epilepsy Seizure Detection using Causal Asymmetries in Deep Learning Models
Code developed by Tenzin Sangpo Choedon, Summer 2025  
Extension on the code developed by Tenzin Sangpo Choedon during [the bachelor thesis study](https://github.com/sappoPrivy/Causal_asymmetries_in_epilepsy_using_CCM)

<!--
## 📄 Abstract
Extension on the code developed by Tenzin Sangpo Choedon during [the thesis study](https://github.com/sappoPrivy/Causal_asymmetries_in_epilepsy_using_CCM)
-->
## 🗂️ Table of Contents

<!--- [Abstract](#-abstract)-->
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Usage](#-usage)
- [Key Findings](#-key-findings)

## 🏗️ Project Structure

root/  
├── src/  # Python scripts for preprocessing, computing and evaluating ccm scores  
├── docs/  # Documentations such as a brief report and select few generated illustrations  
├── data/  # CHB-MIT dataset is loaded and stored here  
├── processed_data/  # Preprocessed data from CHB-MIT dataset is stored here  
├── output_data/  # CCM output of the preprocessed data  
├── graphs_data/  # Graph datasets is stored here  
├── eval_data/  # Evaluation and performance metric values are stored here  
├── README.md  
└── .gitignore  

## ✅ Prerequisites

- **Python 3.7+ packages**
  ```bash
  pip install numpy scipy pandas matplotlib pyEDM
  ```
- **CCM code**  
  Download Python juypiter version from [here](https://phdinds-aim.github.io/time_series_handbook/06_ConvergentCrossMappingandSugiharaCausality/ccm_sugihara.html#introduction)

- **CHB-MIT dataset**  
  Download dataset from [here](https://physionet.org/content/chbmit/1.0.0/#files-panel) and store it in root/data folder 

- **pyTorch packages**  
  ```bash
  pip install tourch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
  ```

- **Surrogate package**  
  ```bash
  pip install neurokit2
  ```

- **Bayesian optimization for hyperparameter tuning**  
  ```bash
  pip install optuna
  ```
  
## 🚀 Usage

**Run automatically all codes and create all directories**
```bash
python src/main.py
```

Otherwise run the scripts in this order and change the global variables accordingly in config.py depending on result analysis:

1. **Preprocess data** (results automatically stored in processed_data)  
   ```bash
   python src/preprocess_data.py
   ```
2. **Compute CCM on subjects** (results automatically stored in output_data)   
   ```bash
   python src/process_CCM_subjects.py
   ```
3. **Generate and test Surrogate data of subjects** (results automatically stored in output_data)  
   ```bash
   python3 src/test_surrogate_data.py
   ```
4. **Evaluate CCM results of all subjects** (results automatically stored in eval_data)  
   ```bash
   python src/eval_CCM_subjects.py
   ```
5. **Classify CCM results of all subjects** (results automatically stored in graphs_data and eval_data)  
   ```bash
   python src/classify_CCM_Graphs.py
   ```

## 📘 Key Findings

<p align="center">
  <img src="docs/Overall-asymmetry-channel-freqs.png" alt="Overall Asymmetry Channel Frequencies" width="70%"/>
</p>

**Performance evaluation on whole causality matrices**
| Metric        | Non-seizure | Pre-seizure | Seizure |
|---------------|-------------|-------------|---------|
| **TP**        | 4           | 6           | 5       |
| **FP**        | 13          | 12          | 14      |
| **FN**        | 14          | 12          | 13      |
| **Sensitivity** | 0.22      | 0.33        | 0.28    |
| **Precision**   | 0.24      | 0.33        | 0.26    |
| **F1-score**    | 0.23      | 0.33        | 0.27    |
<div align="center">
**Performance evaluation on reduced causality matrices**
| Metric       | Non-seizure | Pre-seizure | Seizure |
|--------------|-------------|-------------|---------|
| **TP**       | 9           | 4           | 6       |
| **FP**       | 17          | 7           | 11      |
| **FN**       | 9           | 14          | 12      |
| **Sensitivity** | 0.50     | 0.22        | 0.33    |
| **Precision**   | 0.35     | 0.36        | 0.35    |
| **F1-score**    | 0.41     | 0.28        | 0.34    |
</div>
[The Rapport is in progress](docs/Rapport.pdf)

<!--
<p align="center">
  <img src="docs/Overall-asymmetry-index-distribution.png" alt="Overall Asymmetry Index Distribution" width="70%"/>
</p>
-->

