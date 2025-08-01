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
from config import config

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
def test_subject(subject):
    
    # Start processing subject
    print(f"Starting subject {subject}")
    subject_dir = Path(config.PROC_DIR + "/" + subject)
    
    # Selected patient files
    control_file = os.path.join(subject_dir, "control-data.npz")
    patient_ictal_files = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="ictal"]
    patient_pre_ictal_files = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="pre"]
    
    # Output paths
    output_dir_subj = config.OUTPUT_DIR + '/' + subject
    os.makedirs(output_dir_subj, exist_ok=True)
    output_filename_cs = output_dir_subj + "/control-file-surrogate"
    output_filename_ics = output_dir_subj + '/patient-ictal-file-surrogate'
    output_filename_pres = output_dir_subj + '/patient-pre-ictal-file-surrogate'
    
    # Load control data
    X_c = np.load(control_file)['arr']
    X_ic, ic_len = combine_samples(subject, patient_ictal_files)
    X_pre, pre_len = combine_samples(subject, patient_pre_ictal_files)
    
    # Length decides amount of datapoints in the window
    global end_index
    global start_index
    
    start_index= random.randint(config.OPT_L, X_c.shape[1] - config.OPT_L - 1)
    end_index = start_index + config.OPT_L
    X_cs = generate_surrogates(config.ALL_CHANNELS, X_c)
    compute_across_params([config.OPT_L], [config.OPT_E],[config.OPT_TAU],output_filename_cs, config.ALL_CHANNELS, X_cs)
    
    start_index= random.randint(config.OPT_L, ic_len - config.OPT_L - 1)
    end_index = start_index + config.OPT_L
    X_ics = generate_surrogates(config.ALL_CHANNELS, X_ic)
    compute_across_params([config.OPT_L], [config.OPT_E],[config.OPT_TAU],output_filename_ics, config.ALL_CHANNELS, X_ics)
    
    start_index= random.randint(config.OPT_L, pre_len - config.OPT_TAU - 1)
    end_index = start_index + config.OPT_L
    X_pres = generate_surrogates(config.ALL_CHANNELS, X_pre)
    compute_across_params([config.OPT_L], [config.OPT_E],[config.OPT_TAU],output_filename_pres, config.ALL_CHANNELS, X_pres)
    
    plot_heatmaps(output_dir_subj, config.OPT_L, config.OPT_E, config.OPT_TAU, [output_filename_cs, output_filename_pres, output_filename_ics], config.ALL_CHANNELS, "surrogates-heatmaps")

# CCM Parameters
np.random.seed(1)
L=10000      # length of time period
tau=1       # time lag
E=2         # embedding dimensions

# Select chunk
start_index = 0
end_index = start_index + L

# Generate and test surrogate data of all subjects
def test_surrogates_subjects():

    for subject in config.SELECTED_SUBJECTS:
        test_subject(subject)

if __name__ == "__main__":
    test_surrogates_subjects()
