# Author @Tenzin Sangpo Choedon

import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.interpolate import make_interp_spline
from tqdm import tqdm # for showing progress bar in for loops
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error
### Just to remove warnings to prettify the notebook. 
import warnings
warnings.filterwarnings("ignore")
# import jdc
import multiprocessing as mp
from statsmodels.tsa.stattools import acf
import pyEDM
from process_CCM_subjects import *
from eval_CCM_subjects import *

# Generate surrogate data with random phase using fft
def generate_surrogate(X):
    
    # Apply fourier transform on the signal
    Y = np.fft.fft(X)
    
    # Extract magnitude and phase in fourier transform coefficients
    magnitude = np.abs(Y)
    phase = np.angle(Y)
    
    # Create random phases from 0 to 2pi
    rand_phases = np.random.uniform(0, 2 * np.pi, len(phase))
    
    # Create new fourier coefficients
    new_X = magnitude * np.exp(1j*rand_phases)
    
    # Apply inverse fourier transform
    surrogate = np.fft.ifft(new_X)
    
    return surrogate   

# Generate surrogate data for all channels
def generate_surrogates(limit_channels, X):
    
    # Surrogate matrix
    X_surrogates = np.zeros((len(limit_channels), end_index - start_index))
    
    # Surrogate data for each channel
    for idx, i in enumerate(limit_channels):
        X_surrogates[idx,:] = generate_surrogate(X[i-1,start_index:end_index])
    
    return X_surrogates
        
# Test surrogate data for the subject
def test_subject(subject, proc_data_dir, output_dir):
    
    # Start processing subject
    print(f"Starting subject {subject}")
    subject_dir = Path(proc_data_dir + "/" + subject)
    
    # Selected patient files
    control_file = os.path.join(subject_dir, "control-data.npz")
    patient_ictal_files = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="ictal"]
    patient_pre_ictal_files = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="pre"]
    
    # Output paths
    output_dir_subj = output_dir + '/' + subject
    os.makedirs(output_dir_subj, exist_ok=True)
    output_filename_cs = output_dir_subj + "/control-file-surrogate"
    output_filename_ics = output_dir_subj + '/patient-ictal-file-surrogate'
    output_filename_pres = output_dir_subj + '/patient-pre-ictal-file-surrogate'
    
    # Load control data
    X_c = np.load(control_file)['arr']
    X_ic, ic_len = combine_samples(subject, patient_ictal_files)
    X_pre, pre_len = combine_samples(subject, patient_pre_ictal_files)
    
    # Parameters range
    L_range = [6000, 7000, 8000, 9000, 10000]
    E_range = [2,3, 4, 5]
    tau_range=[1,2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Observed optimal parameter values (L, E, tau) = (10 000, 4, 4)
    opt_L = L_range[4]
    opt_tau = tau_range[3]
    opt_E = E_range[2]
    
    # Length decides amount of datapoints in the window
    global end_index
    global start_index
    
    start_index= random.randint(opt_L, X_c.shape[1] - opt_L - 1)
    end_index = start_index + opt_L
    X_cs = generate_surrogates([i for i in range(1, 24)], X_c)
    compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_cs, [i for i in range(1, 24)], X_cs)
    
    start_index= random.randint(opt_L, ic_len - opt_L - 1)
    end_index = start_index + opt_L
    X_ics = generate_surrogates([i for i in range(1, 24)], X_ic)
    compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_ics, [i for i in range(1, 24)], X_ics)
    
    start_index= random.randint(opt_L, pre_len - opt_L - 1)
    end_index = start_index + opt_L
    X_pres = generate_surrogates([i for i in range(1, 24)], X_pre)
    compute_across_params([opt_L], [opt_E],[opt_tau],output_filename_pres, [i for i in range(1, 24)], X_pres)
    
    plot_heatmaps(output_dir_subj, opt_L, opt_E, opt_tau, [output_filename_cs, output_filename_pres, output_filename_ics], [i for i in range(1, 24)], "surrogates-heatmaps")

# CCM Parameters
np.random.seed(1)
L=10000      # length of time period
tau=1       # time lag
E=2         # embedding dimensions

# Select chunk
start_index = 0
end_index = start_index + L
n_samples = 1


# Get the parent directory
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Define the relative paths
proc_data_dir = os.path.join(parent_dir, 'processed_data')
dummy_data_dir = os.path.join(parent_dir, 'dummy_data')
data_dir = os.path.join(parent_dir, os.path.join('data', "chbmit-1.0.0.physionet.org"))
output_dir = os.path.join(parent_dir, 'output_data')

# EXCLUDED subjects 19, 7, 18, 11, 24
list_subjects = [f"chb{str(i).zfill(2)}" for i in [1, 2, 3,4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23]]
# num_cores = mp.cpu_count()
args_list = [(subject, proc_data_dir, output_dir) for subject in list_subjects]

for subject in list_subjects:
    test_subject(subject, proc_data_dir, output_dir)
