# Author @Tenzin Sangpo Choedon

import os
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain
import torch
import torch_geometric
print(torch_geometric.__version__)
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# Consists of EEG graphs
dataset = []

### Defining the GNN model with 2 GCN layers, ReLu
class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    # Applies layers sequentially on feature data and edge information
    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = global_mean_pool(x, batch)  # Pool over nodes to get graph representation
        return self.lin(x)

### Train the GNN model on the dataset
def train_model(loader):
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
def evaluate_model(loader):
    model.eval()
    
    # Initialize dictionaries
    tp = {0: 0, 1: 0, 2: 0}
    fp = {0: 0, 1: 0, 2: 0}
    fn = {0: 0, 1: 0, 2: 0}
    
    # Test each batch of graphs
    with torch.no_grad():
        for data in loader:
            
            # Model predictions
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            # Predicted labels
            pred = output.argmax(dim=1)
            
            # Compute the successful and unsuccesful detections
            for p, y in zip(pred, data.y):
                if p == y:
                    tp[int(y)] += 1     # Correct detections
                else:
                    fp[int(p)] += 1     # Wrong detections
                    fn[int(y)] += 1     # Missed detections
                    
    return tp, fp, fn

### Creating list of Graphs as input for GNN
def transform_causal_graph(subject, output_data_dir):
    
    # Start processing subject
    print(f"Starting subject {subject}")
    subject_dir = Path(output_data_dir + "/" + subject)
    
    # Selected patient files
    control_file = os.path.join(subject_dir, "control-file.npz")
    patient_ictal_file = os.path.join(subject_dir, "patient-ictal-file.npz")
    patient_pre_ictal_file = os.path.join(subject_dir, "patient-pre-ictal-file.npz")
    
    # Output paths
    output_dir_subj = output_data_dir + '/' + subject
    os.makedirs(output_dir_subj, exist_ok=True)
    output_filename_graphs = output_dir_subj + "/Graphs"
    
    # Parameters range
    L_range = [6000, 7000, 8000, 9000, 10000]
    E_range = [2, 3, 4, 5]
    tau_range=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Observed optimal parameter values (L, E, tau) = (10 000, 4, 4)
    opt_L = L_range[4]
    opt_tau = tau_range[3]
    opt_E = E_range[2]
    
    # Load control data
    C_c = np.load(control_file)
    C_ic = np.load(patient_ictal_file)
    C_pre = np.load(patient_pre_ictal_file)
    
    data_c = C_c[f'L{opt_L}_E{opt_E}_tau{opt_tau}']
    data_ic = C_ic[f'L{opt_L}_E{opt_E}_tau{opt_tau}']
    data_pre = C_pre[f'L{opt_L}_E{opt_E}_tau{opt_tau}']
    mx = [data_c, data_pre, data_ic]
    
    # Identity matrix with 23 channels
    node_features = torch.eye(23)

    # Include all causal relationships as edges except self loops
    edge_index = torch.tensor([(i, j) for i in range(23) for j in range(23) if i != j], dtype=torch.long).T 
    
    for x, label in zip(mx, [0, 1, 2]):
                
        # Include all causality scores as edge weights
        edge_attr = torch.tensor(x[~np.eye(23, dtype=bool)], dtype=torch.float)
        
        data = Data(
            x=node_features, 
            edge_index=edge_index, 
            edge_attr = edge_attr,
            y=torch.tensor([label], dtype=torch.long)
            )
        dataset.append(data)
        print("Added")

# Get the parent directory
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Define the relative paths
output_dir = os.path.join(parent_dir, 'output_data')
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    
    # EXCLUDED subjects 19, 7, 18, 11, 24, 22
    list_subjects = list(reversed([f"chb{str(i).zfill(2)}" for i in [1, 2, 3,4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 20, 21, 23]]))

    for subject in list_subjects:
        transform_causal_graph(subject, output_dir)
        
    # Initialize the GNN model
    num_channels = 23   # number of channels
    num_states = 3      # number of classifications
    model = GraphClassifier(num_channels, 16, num_states)
    
    # Adam optimizer to train model with specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    
    # The number of correct detections over all subjects for each class
    count_tp = {0: 0, 1: 0, 2: 0}
    count_fp = {0: 0, 1: 0, 2: 0}
    count_fn = {0: 0, 1: 0, 2: 0}

    # Utilize leave-one-subject-out cross validation method
    for idx, i in enumerate(list_subjects):
        print(f"LOO round: {idx}")
        
        # Partition the dataset into training and testing sets
        train_dataset = list(chain(dataset[:idx*3], dataset[(idx+1)*3:]))
        test_dataset = dataset[idx*3:(idx+1)*3]
                
        # Train the data
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        loss_value = train_model(train_loader)
        print(f'Training for subject: {i}, Loss: {loss_value:.4f}')
        
        # Test the data
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        tp, fp, fn = evaluate_model(test_loader)
        
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
        sensitivity[c] = count_tp[c] / (count_tp[c] + count_fn[c])
        precision[c] = count_tp[c] / (count_tp[c] + count_fp[c])
        f1_score[c] = 2*sensitivity[c]*precision[c] / (sensitivity[c] + precision[c])
        metrics[c] = [count_tp[c], count_fp[c], count_fn[c], sensitivity[c], precision[c], f1_score[c]]
    
    # Save the metric data
    df = pd.DataFrame(metrics).rename(
        columns={0: "Non-seizure", 1: "Pre-seizure", 2: "Seizure"}, 
        index={0: "TP", 1: "FP", 2: "FN", 3: "Sensitivity", 4: "Precision", 5: "F1-score"}
        )
    df.to_excel(output_dir+"/detection_evaluation_metrics.xlsx")
    
    
    
        
