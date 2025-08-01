# Author @Tenzin Sangpo Choedon

from functools import partial
import logging
import os
import random
import re
import sys
import mne
from mne.time_frequency import psd_array_multitaper
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fooof import FOOOF, FOOOFGroup
import time
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from config import config
from fooof import FOOOF, FOOOFGroup
from process_CCM_subjects import combine_samples

# Find and collect the annotations of relevant non seizure and seizure samples of a subject
def collect_subject_samples(subject):
    
    # Subject specific dirs
    print(f"Extracting annotations of subject: {subject}\n")
    annotations_dir = os.path.join(config.DATA_DIR, subject, subject+"-summary.txt")
    
    # Initialize the collection of samples for the subject
    subject_samples = {
        "subject_name": subject,
        "control_file": "",
        "seizure_samples": []
    }
    findFile = False
    
    # Localize the seizure samples and a control file
    with open(annotations_dir, "r") as f:
        for l in f:
            
            # Register filename and initialize variables
            if l.strip().split(": ")[0] == "File Name": 
                filename = l.strip().split(": ")[1]
                print(filename)
                findFile = True
                num_seizure = 0
                seizure_onsets = [] 
                seizure_offsets = []
                continue
            
            # Process the samples of the filename
            if findFile:
                if l.strip() == '': 
                    
                    if num_seizure == 0:
                        if not subject_samples['control_file']:
                            subject_samples['control_file'] = filename
                            print(f"Control file: {filename}")
                    else:                          
                        seizure_sample = {
                            "filename": filename,
                            "num_seizures": num_seizure,
                            "onsets": seizure_onsets,
                            "durations":[offset-onset for onset, offset in zip(seizure_onsets, seizure_offsets)],
                            "descriptions": [("seizure_"+ str((i+1))) for i in range(len(seizure_onsets))]
                        }
                        print(seizure_sample)
                        subject_samples['seizure_samples'].append(seizure_sample)
                    findFile= False;    
                if l.strip().split(": ")[0] == "Number of Seizures in File":
                    num_seizure = int(l.strip().split(": ")[1])
                if l.startswith("Seizure") and "Start Time:" in l:
                    seizure_onsets.append(float(l.split(":")[1].strip().split(" ")[0]))
                if l.startswith("Seizure") and "End Time:" in l:
                    seizure_offsets.append(float(l.split(":")[1].strip().split(" ")[0]))
                    
    return subject_samples

# Preprocess the data of a subject
def preprocess_subject(subject):
    
    # Subject specific dirs
    print(f"Preprocessing subject {subject}")
    data_dir_subj = os.path.join(config.DATA_DIR,subject)
    proc_dir_subj = os.path.join(config.PROC_DIR, subject)
    os.makedirs(proc_dir_subj, exist_ok=True)
    
    num_ses=0
    num_pre_ses=0
    
    # Preparing subject for segmenting into seizure, pre-seizure samples
    subject_samples = collect_subject_samples(subject)
        
    # Load and preprocess the non-seizure sample
    control_data = mne.io.read_raw_edf(os.path.join(data_dir_subj, subject_samples['control_file']), preload = True)
    control_data.filter(l_freq=0.5, h_freq=40)
    proc_filename_c = proc_dir_subj + '/control-data'
    np.savez_compressed(file=proc_filename_c, arr=control_data.get_data())
    
    # Running through all the files containing seizures
    for sample in subject_samples['seizure_samples']:
                
        # Loading the file containing seizure
        patient_data = mne.io.read_raw_edf(os.path.join(data_dir_subj, sample['filename']), preload = True)
        
        patient_data.filter(l_freq=0.5, h_freq=40)
        sfreq = patient_data.info['sfreq']
                
        # Loop through each seizure and crop the data
        for i in range(sample['num_seizures']):
            # Convert annotation start and stop time (in seconds)
            start_time = sample['onsets'][i]
            end_time = start_time + sample['durations'][i]
            
            # Crop the data around the seizure
            start_sample = int(start_time * sfreq)
            stop_sample = int(end_time * sfreq)
            
            # Crop the data around the seizure
            seizure_data = patient_data.get_data(start=start_sample, stop=stop_sample)
            
            num_ses += 1
            proc_filename = proc_dir_subj + '/ictal-data-' + str(num_ses)
            
            # Save the seizure data as a .npy file
            np.savez_compressed(file=proc_filename, arr=seizure_data)
            
            print("Saved seizure to "+proc_filename)
            
            # Crop the data around the pre seizure
            start_pre_sample = int((start_time-30) * sfreq)
            stop_pre_sample = int(start_time * sfreq)
            
            # Crop the data around the pre seizure
            pre_seizure_data = patient_data.get_data(start=start_pre_sample, stop=stop_pre_sample)
            
            num_pre_ses += 1
            proc_pre_filename = proc_dir_subj + '/pre-ictal-data-' + str(num_pre_ses)
            
            # Save the pre seizure data as a .npy file
            np.savez_compressed(file=proc_pre_filename, arr=pre_seizure_data)
            
            print("Saved pre seizure to "+proc_pre_filename)

# Compute and save aperiodic slope of data range
def compute_aperiodic_slope(X, filename):
    lambda_matrix = []
    segment_size = 512 # 2 seconds
    step = 256  # Overlapping 50%
    freq = 256  # Hz
    psds_list = []
    print(f"Shape of X: {X.shape}")
    # Segmenting into 2 seconds
    for start in range(0, X.shape[1] - segment_size + 1, step):
        segment = X[:, start:start + segment_size]
        psds, freqs = psd_array_multitaper(segment, sfreq=freq)
        psds = np.maximum(psds, 1e-12)      # Ensure no zeros or negatives
        psds_list.append(psds)
    
    # Computing aperiodic slope on all psds values
    all_psds = np.stack(psds_list)
    psds_reshape = all_psds.reshape(-1, all_psds.shape[-1])
    fm = FOOOFGroup()
    start_time = time.time()
    fm.fit(freqs, psds_reshape)
    exps = fm.get_params('aperiodic_params', 'exponent')
    lambda_matrix = exps.reshape(all_psds.shape[:2]) # (segments, channels)
    end_time = time.time()
        
    print(f"Time for computing exponent: {end_time - start_time:.4f} seconds\n")
    print(f"Shape of lambda matrix {lambda_matrix.shape}")
    np.save(filename, lambda_matrix)

# Extract features such as aperiodic slope
def extract_additional_features(subject):
    
    # Start extracting subject
    print(f"Starting subject {subject}")
    subject_dir = Path(config.PROC_DIR + "/" + subject)
    
    # Selected patient files
    control_data = os.path.join(subject_dir, "control-data.npz")
    patient_ictal_data = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="ictal"]
    patient_pre_ictal_data = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f)) and f.split("-")[0]=="pre"]
    
    # Output paths
    filename_c = os.path.join(subject_dir, "control-data-aperiodic")
    filename_ic = os.path.join(subject_dir, 'patient-ictal-data-aperiodic')
    filename_pre = os.path.join(subject_dir, 'patient-pre-ictal-data-aperiodic')
    
    # Load preprocessed data
    X_c = np.load(control_data)['arr']
    X_ic, ic_len = combine_samples(subject, patient_ictal_data)
    X_pre, pre_len = combine_samples(subject, patient_pre_ictal_data)
        
    start_index= random.randint(config.OPT_L, X_c.shape[1] - config.OPT_L - 1)
    end_index = start_index + config.OPT_L
    X_c = X_c[:,start_index:end_index]
    
    start_index= random.randint(config.OPT_L, ic_len - config.OPT_L - 1)
    end_index = start_index + config.OPT_L
    X_ic = X_ic[:,start_index:end_index]
    
    start_index= random.randint(config.OPT_L, pre_len - config.OPT_L - 1)
    end_index = start_index + config.OPT_L
    X_pre = X_pre[:,start_index:end_index]
    mx = [X_c, X_pre, X_ic]
    filenames = [filename_c, filename_pre, filename_ic]
    
    # Extract feature for each state
    for state in range(config.NUM_STATES):
        compute_aperiodic_slope(mx[state], filenames[state])

# Preprocessing executed on all subjects
def preprocess():
    
    # Preprocess all subjects
    for subject in config.SELECTED_SUBJECTS[:]:
        preprocess_subject(subject)
        extract_additional_features(subject)
    
    # Preprocess only one subject
    # preprocess_subject(config.TEST_SUBJECT)
    # extract_additional_features(config.TEST_SUBJECT)

if __name__ == "__main__":
    preprocess()
