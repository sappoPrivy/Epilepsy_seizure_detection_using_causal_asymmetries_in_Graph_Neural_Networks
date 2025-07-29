# Author @Tenzin Sangpo Choedon

import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain
import torch
import torch_geometric
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader
import optuna
from config import config

# Consists of EEG graphs
dataset = []

### Defining the GNN model with 2 GCN layers, ReLu
class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphClassifier, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=1)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, edge_dim=1)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    # Applies layers sequentially on feature data and edge information
    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight.unsqueeze(-1)))        # Ensure edge weigths is [num_edges, 1]
        x = F.relu(self.conv2(x, edge_index, edge_weight.unsqueeze(-1)))
        x = global_mean_pool(x, batch)                                          # Pool over nodes to get graph representation
        return self.lin(x)

### Train the GNN model on the dataset
def train_model(model, optimizer, loader):
    model.train()
    tot_loss = 0
    for data in loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    return tot_loss / len(loader)

### Evaluating the Model's Performance
def evaluate_model(model, loader):
    model.eval()
    
    # Initialize dictionaries
    tp = {0: 0, 1: 0, 2: 0}
    fp = {0: 0, 1: 0, 2: 0}
    fn = {0: 0, 1: 0, 2: 0}
    
    tot_samples = 0
    tot_correct = 0
    
    # Test each batch of graphs
    with torch.no_grad():
        for data in loader:
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = output.argmax(dim=1)
            
            for p, y in zip(pred, data.y):
                tot_samples += 1
                if p == y:
                    tp[int(y)] += 1     # Correct detections
                    tot_correct += 1
                else:
                    fp[int(p)] += 1     # Wrong detections
                    fn[int(y)] += 1     # Missed detections
    accuracy = tot_correct / tot_samples if tot_samples > 0 else 0.0
    return tp, fp, fn, accuracy

def objective_wrapper(val_dataset, list_subjects):
    def objective(trial):
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        # Initialize the GNN model
        model = GraphClassifier(config.NUM_CHANNELS, 16, config.NUM_STATES)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        tot_accuracy = 0
        
        # Utilize leave-one-subject-out cross validation method
        for idx, i in enumerate(list_subjects):
            print(f"Inner Fold: {idx}")

            # Partition the training dataset for validation
            train_dataset = list(chain(val_dataset[:idx*config.NUM_STATES], val_dataset[(idx+1)*config.NUM_STATES:]))
            test_dataset = val_dataset[idx*config.NUM_STATES:(idx+1)*config.NUM_STATES]
        
            # Train the data
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            loss_value = train_model(model, optimizer, train_loader)
            print(f'Training for subject: {i}, Loss: {loss_value:.4f}')
            
            # Test the data
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            tp, fp, fn, accuracy = evaluate_model(model, test_loader)
            print(f'Testing for subject: {i}, Accuracy: {accuracy}')
            
            tot_accuracy += accuracy
            
        return tot_accuracy / len(list_subjects)
    return objective

### Creating list of Graphs as input for GNN
def transform_causal_graph(subject):
    
    # Start processing subject
    print(f"Starting subject {subject}")
    subject_dir = Path(config.OUTPUT_DIR + "/" + subject)
    
    # Selected patient files
    control_file = os.path.join(subject_dir, "control-file.npz")
    patient_ictal_file = os.path.join(subject_dir, "patient-ictal-file.npz")
    patient_pre_ictal_file = os.path.join(subject_dir, "patient-pre-ictal-file.npz")
        
    # Load control data
    C_c = np.load(control_file)
    C_ic = np.load(patient_ictal_file)
    C_pre = np.load(patient_pre_ictal_file)
    
    data_c = C_c[f'L{config.OPT_L}_E{config.OPT_E}_tau{config.OPT_TAU}']
    data_ic = C_ic[f'L{config.OPT_L}_E{config.OPT_E}_tau{config.OPT_TAU}']
    data_pre = C_pre[f'L{config.OPT_L}_E{config.OPT_E}_tau{config.OPT_TAU}']
    mx = [data_c, data_pre, data_ic]
    
    # Identity matrix with 23 channels
    node_features = torch.eye(config.NUM_CHANNELS)

    # Include all causal relationships as edges except self loops
    edge_index = torch.tensor([(i, j) for i in range(config.NUM_CHANNELS) for j in range(config.NUM_CHANNELS) if i != j], dtype=torch.long).T 
    
    for x, label in zip(mx, config.LABELS):
                
        # Include all causality scores as edge weights
        x = x[~np.eye(config.NUM_CHANNELS, dtype=bool)]             # Upper and lower triangle with no self-loops
        x = (x - np.mean(x)) / (np.std(x) + 1e-6)                   # Normalized causality values
        edge_attr = torch.tensor(x, dtype=torch.float)
        
        data = Data(
            x=node_features, 
            edge_index=edge_index, 
            edge_attr = edge_attr,
            y=torch.tensor([label], dtype=torch.long)
            )
        dataset.append(data)
        print("Added")

# Classify states using data from all subjects
def classify_states():
    
    for subject in config.SELECTED_SUBJECTS:
        transform_causal_graph(subject)
        torch.save(dataset, config.GRAPHS_DIR+"/dataset.pt")
        
    # The number of correct detections over all subjects for each class
    count_tp = {0: 0, 1: 0, 2: 0}
    count_fp = {0: 0, 1: 0, 2: 0}
    count_fn = {0: 0, 1: 0, 2: 0}

    # Utilize nested cross validation method
    for idx, i in enumerate(config.SELECTED_SUBJECTS):
        print(f"Outer Fold: {idx}")
        
        # Partition the dataset into training and testing sets
        train_dataset = list(chain(dataset[:idx*config.NUM_STATES], dataset[(idx+1)*config.NUM_STATES:]))
        test_dataset = dataset[idx*config.NUM_STATES:(idx+1)*config.NUM_STATES]
        
        # Hyper parameter tuning
        tune = optuna.create_study(direction="maximize")
        tune.optimize(objective_wrapper(train_dataset, config.SELECTED_SUBJECTS[:idx] + config.SELECTED_SUBJECTS[idx+1:]), n_trials=100)
        best_params = tune.best_params
        
        # Adam optimizer to train model with learning rate and weight decay
        model = GraphClassifier(config.NUM_CHANNELS, 16, config.NUM_STATES)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        
        # Train the data
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        loss_value = train_model(model, optimizer, train_loader)
        print(f'Training for subject: {i}, Loss: {loss_value:.4f}')
        
        # Test the data
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        tp, fp, fn, _ = evaluate_model(model, test_loader)
        
        # Update counters
        for k in tp.keys():
            count_tp[k] += tp[k]
        for k in fp.keys():
            count_fp[k] += fp[k]
        for k in fn.keys():
            count_fn[k] += fn[k]
        
        print(f'Testing for subject: {i}')
    
    # Metrics for all classes
    metrics = {}
    sensitivity = {}
    precision = {}
    f1_score = {}
    
    # Compute performance metrics for each class
    for c in count_tp.keys():
        sensitivity[c] = count_tp[c] / (count_tp[c] + count_fn[c]) if (count_tp[c] + count_fn[c]) > 0 else 0
        precision[c] = count_tp[c] / (count_tp[c] + count_fp[c]) if (count_tp[c] + count_fp[c]) > 0 else 0
        f1_score[c] = 2*sensitivity[c]*precision[c] / (sensitivity[c] + precision[c]) if (sensitivity[c] + precision[c]) > 0 else 0
        metrics[c] = [count_tp[c], count_fp[c], count_fn[c], sensitivity[c], precision[c], f1_score[c]]
    
    # Save the metric data
    df = pd.DataFrame(metrics).rename(
        columns=config.STATES, 
        index={0: "TP", 1: "FP", 2: "FN", 3: "Sensitivity", 4: "Precision", 5: "F1-score"}
        )
    df.to_excel(config.EVAL_DIR+"/detection_evaluation_metrics_hyper_2.xlsx")
    
