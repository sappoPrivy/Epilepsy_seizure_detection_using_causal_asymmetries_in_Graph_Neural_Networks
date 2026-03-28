# Epilepsy Seizure Detection using Causal Asymmetries in Deep Learning Models
Code developed by Tenzin Sangpo Choedon, Summer 2025  
Extension on the code developed by Tenzin Sangpo Choedon during [the bachelor thesis study](https://github.com/sappoPrivy/Causal_asymmetries_in_epilepsy_using_CCM)

## Project Overview
The end goal of this project is to classify EEG data into three categories: non-seizure, pre-seizure and seizure state, using CCM matrices, then perform a comparison between employing i) all 23 EEG channels present in the CCM matrix and ii) selecting EEG channels that show most consistently dominant asymmetric patterns in the CCM matrix.
Models implemented:
- Binary GNN classification of seizure vs non-seizure state with all 23 EEG channels.
- Multi-state GNN classification of pre-seizure, seizure vs non-seizure state with all 23 EEG channels.
- Multi-state GNN classification of pre-seizure, seizure vs non-seizure state with reduced EEG channels.
<!-- 
The purpose of this project is to classify EEG data into three categories: non-seizure, pre-seizure and seizure state, using CCM matrices, then perform a comparison between employing i) all 23 EEG channels present in the CCM matrix and ii) selecting EEG channels that show most consistently dominant asymmetric patterns in the CCM matrix. This is achieved using a GNN model, thus this becomes a graph classification problem. The graph (G) is composed of nodes that represent EEG channels, set of edges that represent causal relationships between EEG channels and edge weights that represent the causality (CCM) scores of the relationship. The goal is to learn the function: f(G) -> y, where it predicts a class label given an input graph G. The class labels 0, 1, 2 corresponds to non-seizure, pre-seizure and seizure state.  However, the F1-score of the baseline model i), as well as the F1-score for the reduced channel model ii), are very low and show no significant difference, indicating that the model does not particularly learn anything. In order to improve the performance of the models, the causality scores were normalized, Transformer GNN was implemented and hyperparamter tuning was performed for learning rate, weight decay, number of GCN layers, hidden layer size, dropout and batch size using Bayesian Optimization. Additionally, node features were included - 38 aperiodic slope values per node, and these were also normalized. Although several methods has been utilized, it did not improve the performance of the models. These results could be due to the limited dataset leading to the model underfitting or due to lack of meaningful data. Therefore, the result is inconclusive, though there could still be value in this project since further enhancements or extended datasets could solve this issue. **This is still a work in progress and may have potential updates**.
-->
## Project Structure
```bash
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
```

## 📦 Prerequisites

- **Install Dependencies**
  ```bash
  pip install numpy scipy pandas matplotlib pyEDM tourch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric neurokit2 optuna
  ```
- **CCM code**  
  Download Python juypiter version from [here](https://phdinds-aim.github.io/time_series_handbook/06_ConvergentCrossMappingandSugiharaCausality/ccm_sugihara.html#introduction)

- **CHB-MIT dataset**  
  Download dataset from [here](https://physionet.org/content/chbmit/1.0.0/#files-panel) and store it in root/data folder 
  
## 🚀 Usage

**Run the following**

1. **Preprocess data** (partitioning, preprocessing and computing CCM)  
   ```bash
   python src/preprocess_data_sliding.ipynb
   ```
2. **Classify data** (classify and evaluate the CCM matrices)   
   ```bash
   python src/binary_GNN.ipynb
   ```
<!-- 
```bash
python src/main.py
```

Otherwise run the scripts in this order and change the global variables accordingly in config.py depending on result analysis:
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

-->
   
## Model Performance Summary

**Performance evaluation on Model i)  whole causality matrices**  
| Metric       | Non-seizure | Pre-seizure | Seizure |
|--------------|-------------|-------------|---------|
| TP           | 6           | 7           | 7       |
| FP           | 12          | 10          | 12      |
| FN           | 12          | 11          | 11      |
| Sensitivity  | 0.33        | 0.39        | 0.39    |
| Precision    | 0.33        | 0.41        | 0.37    |
| F1-score     | 0.33        | 0.40        | 0.38    |

**Performance evaluation on Model ii) reduced causality matrices**  
| Metric       | Non-seizure | Pre-seizure | Seizure |
|--------------|-------------|-------------|---------|
| TP           | 5           | 13          | 1       |
| FP           | 8           | 25          | 2       |
| FN           | 13          | 5           | 17      |
| Sensitivity  | 0.28        | 0.72        | 0.06    |
| Precision    | 0.38        | 0.34        | 0.33    |
| F1-score     | 0.32        | 0.46        | 0.10    |

Note: However, the F1-score of the baseline model i), as well as the F1-score for the reduced channel model ii), are very low and show no significant difference, indicating that the model does not particularly learn anything (underfitting), which can be largely attributed to the limited amount of CCM matrices.

**Performance evaluation on Binary Model with Overlapping Windows**  
| Metric       | Non-seizure |  Seizure |
|--------------|-------------|---------|
| TP           | TBD         | TBD       |
| FP           | TBD           | TBD       |
| FN           | TBD          | TBD      |
| Sensitivity  | TBD        | TBD    |
| Precision    | TBD        | TBD    |
| F1-score     | TBD        | TBD    |

Note: Using overlapping sliding windows for the EEG data, allows us to compute more CCM matrices; thereby, increasing the sample size significantly (sufficiently for the model to learn).

<!--

<p align="center">
  <img src="docs/GNN Architecture.png" alt="Architecture of GNN model" width="70%"/>
</p>

<p align="center">
  <img src="docs/Overall-asymmetry-channel-freqs.png" alt="Overall Asymmetry Channel Frequencies" width="70%"/>
</p>
<p align="center">
  <img src="docs/Overall-asymmetry-index-distribution.png" alt="Overall Asymmetry Index Distribution" width="70%"/>
</p>
-->
## Documentation 🔗
Inside the [documentation](https://github.com/sappoPrivy/Epilepsy_seizure_detection_using_causal_asymmetries_in_Graph_Neural_Networks/tree/main/docs) folder you will find:
- **methods.md** - methodology, preprocessing, training pipeline
