# Author @Tenzin Sangpo Choedon

import logging
import os
from pathlib import Path
import random
from typing import Counter
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
import statistics
### Just to remove warnings to prettify the notebook. 
import warnings
warnings.filterwarnings("ignore")
# import jdc
import time
import multiprocessing as mp
from statsmodels.tsa.stattools import acf
import pyEDM
from config import config

# Plot heatmap for single causality matrix
def plot_heatmap(L, E, tau, output_filename, limit_channels, type):
    
    # Load all causality matrices
    X = np.load(output_filename+'.npz')
    
    # The selected causality matrix
    data = X[f'L{L}_E{E}_tau{tau}']
    
    # Create figure and heatmap
    plt.figure(figsize=(10, 8))
    channel_arr = [f"Ch{i}" for i in limit_channels]
    sns.heatmap(data, annot=True, cmap="coolwarm", square=True, vmin=0,vmax=1, xticklabels=channel_arr, yticklabels=channel_arr)
    
    # Plot the causality matrix as a heatmap
    plt.title(f'{type} correls heatmaps for L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_filename+f'-heatmap.png')
    plt.close()

# Plot heatmaps for multiple causality matrices
def plot_heatmaps(output_subj_dir, L, E, tau, output_filenames, limit_channels, title):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    channel_arr = [f"Ch{i}" for i in limit_channels]
    states=["a) Non-seizure", "b) Pre-ictal", "c) Ictal"]
    
    for idx, (s, output_filename) in enumerate(zip(states, output_filenames)):
        # Load all causality matrices
        X = np.load(output_filename+'.npz')
        
        # Current causality matrix
        data = X[f'L{L}_E{E}_tau{tau}']
            
        # Heatmap of the causality matrix
        sns.heatmap(data, annot=False, cmap="coolwarm", square=True,vmin=0,vmax=1,xticklabels=channel_arr, yticklabels=channel_arr, ax=axs[idx], cbar=False)
            
        axs[idx].set_title(f"{s} state")
    
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(axs[2].get_children()[0], cax=cbar_ax)
    cbar.set_label('Causality', fontsize=12)
    
    fig.suptitle(f'Causality heatmaps with L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_subj_dir+f"/{output_subj_dir.split('/')[-1]}-{title}.png")
    plt.close()

# Plot distribution of asymmetry index values as boxplot
def plot_boxplot_asymm(asymm_idx_subjects, output_dir, L, E, tau):
    
    plt.figure(figsize=(10, 8))
    
    # Load the asymmetry index from the list of dictionaries
    c = np.array([dict['control'] for dict in asymm_idx_subjects])
    pre = np.array([dict['preictal'] for dict in asymm_idx_subjects])
    ic = np.array([dict['ictal'] for dict in asymm_idx_subjects])
    mx = [c, pre, ic]
    
    # Plot the distribution
    plt.boxplot(mx, labels=["Control", "Pre-ictal", "Ictal"], showfliers=False)
    plt.ylabel('Asymmetry Index')
    
    # Customize spacing in and between groups
    xs = []
    
    # 10 unique colors
    colors = plt.cm.tab10.colors

    # Add x with spacings
    for idx, state in enumerate(mx, start=1):
        x = np.random.normal(loc=idx, scale=0.05, size=len(state))
        xs.append(x)
    
    # Plot for each subject
    for i in range(len(mx[0])):
        
        # All x points across states
        x_s = [xs[idx][i] for idx in range(3)]
        
        # All asymmetry index value for same subject across states
        y_s = [mx[idx][i] for idx in range(3)]
        
        # Choose unique color within each cycle of 10 subjects
        unique_color = colors[i % len(colors)]
        
        # Plot the points and corresponding lines between
        plt.plot(x_s, y_s, 'o', color=unique_color, alpha=1, markersize=8)
        plt.plot(x_s, y_s, color=unique_color, linestyle='-', alpha=0.7, linewidth=2)
    
    # Statistics text under figure
    ymin, _ = plt.ylim()
    text_y = ymin - 0.07 * (plt.ylim()[1] - ymin)

    # Add statistics under each state
    for idx, state in enumerate(mx, start=1):
        mean = np.mean(state)
        sd = np.std(state)
        median = np.median(state)
        iqr = np.percentile(state, 75) - np.percentile(state, 25)
        text=(f"Mean={mean:.2f}\nMedian={median:.2f}\nIQR={iqr:.2f}\nSD={sd:.2f}")
        plt.text(idx, text_y, text, ha='center', va='top', fontsize=10)

    plt.title(f'Asymmetry Index Distributions L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir+ f'/Overall-asymmetry-index-distribution.png')
    plt.close()

# Remove the rows and columns that contain Nan in the causality matrix
def cleanMatrix(m):
    
    i=0
    # Check entries with Nan
    while i < m.shape[0] :

        # First entry contains Nan
        if not (np.isfinite(m[i, 0]) and np.isfinite(m[0, i])):
            m = np.delete(m, i, 0)  # Remove row i
            m = np.delete(m, i, 1)  # Remove column i
        
        # Row contains no Nans
        else:
            i += 1
            
    return m
    

# Compute asymmetry index for each causality matrix in the different states
def compute_asymm_idx(subject, output_filenames, L, E, tau):
    
    # List of asymmetry index values
    asymm_idxs=[]
    
    # Compute asymm idx for each state
    for output_filename in output_filenames:
        
        # Load and clean causality matrix
        X = np.load(output_filename+'.npz')
        data = X[f'L{L}_E{E}_tau{tau}']
        data = cleanMatrix(data)
                
        # Calculate asymm index for the causality matrix
        asymm_idx = np.linalg.norm(data - data.T, 'fro')
        asymm_idxs.append(asymm_idx)
        print(f'Done computing asymmetry index for {output_filename}\n')
    
    # Dictionary of asymmetry index values for all state
    asymm_row = {'subject_id': subject, 'control': asymm_idxs[0], 'preictal': asymm_idxs[1], 'ictal': asymm_idxs[2]}
    print(f'Done computing asymmetry index for patient specific files {subject}\n')
    return asymm_row

# Identify the top_N channels with the higher asymmetry value for each state
def compute_ch_asymm(subject, output_filenames, L, E, tau, top_N, format=False):

    # List of channels with max asymmetry for each state
    max_asymm_ch_states=[]    
    
    # Find highest asymmetry for each state
    for output_filename in output_filenames:
        
        # Load causality matrix
        X = np.load(output_filename+'.npz')
        data = X[f'L{L}_E{E}_tau{tau}']
        
        # List of channels and corresponding asymmetry value
        asymm_ch_list = []
        for i in range(data.shape[0]-1):
            
            # Total causal outflow
            outF = np.sum(data[i,:])
            
            # Total causal inflow
            inF = np.sum(data[:,i])
            
            # Asymmetry value is quantified with the difference between outflow and inflow
            asymm = np.abs(outF-inF)
            asymm_ch_list.append({'key': f'{i+1}', 'value': asymm})
        
        # Get top N channel and value pair with highest asymmetry value
        max_asymm_ch = sorted(asymm_ch_list, key=lambda x: x['value'], reverse=True)[:top_N]
        
        # Select this dominant channel for the state
        if format and top_N==1:
            max_asymm_ch_states.append(f"Ch({max_asymm_ch[0]['key']}): {round(max_asymm_ch[0]['value'], 3)}")
            print(max_asymm_ch)
            print(f'Done computing asymmetry channel for {output_filename}\n')
        else:
            max_asymm_ch_states.append(max_asymm_ch)
            print(max_asymm_ch_states)
            print(f'Done computing {top_N} asymmetry channel for {output_filename}\n')
    
    # Dictionary of the total dominant asymmetric channels for each state
    asymm_row={'subject_id': subject, 'control': max_asymm_ch_states[0], 'preictal': max_asymm_ch_states[1], 'ictal': max_asymm_ch_states[2]}
    print(f'Done finding dominant asymmetric channel for patient specific files {subject}\n')
    return asymm_row

# Plot the top 3 frequent dominant asymmetric channels for each state
def plot_frequency_asymm_ch(asymm_subjects, output_dir, L, E, tau):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))  # 1 row, 3 columns
    
    # Create horisontal bar chart of frequency for channel pair for each state
    for idx, state in enumerate(['control', 'preictal', 'ictal']):
        
        # Get the channels with highest asymmetry for that state
        # state_dom_ch = np.array([dict[state] for dict in asymm_subjects])
        state_dom_ch = np.array([in_dict for o_dict in asymm_subjects for in_dict in o_dict[state]])
        freq_dom_ch = []
        print(state_dom_ch)
        
        # Compute the frequency of the channel pair
        for pair in state_dom_ch:
            print(type(pair))
            ch = pair['key']
            asymm_value = pair['value']
            exist = False
            
            # Check if the pair is already registered
            for p in freq_dom_ch:
                if p['ch'] == f'{ch}':
                    p['freq'] += 1
                    p['sum'] += asymm_value
                    exist = True
                    break
                
            # Register the channel pair the first time
            if not exist:
                freq_dom_ch.append({'ch': f'{ch}', 'freq': 1,'sum': asymm_value})
        
        # Compute the top 3 most frequently occuring channels
        top_3 = sorted(freq_dom_ch, key=lambda x: x['freq'], reverse=True)[:3]
        
        y = np.array([dict['ch'] for dict in top_3])
        x = np.array([dict['freq'] for dict in top_3])
        
        # Plot the horisontal bar charts for the state
        bars=axs[idx].barh(y, x, color='skyblue')
        axs[idx].set_xlabel('Frequency')
        axs[idx].set_title(f'{state.capitalize()} state')
        axs[idx].set_yticks(np.arange(len(y)))
        axs[idx].set_yticklabels(y)
        
        # Display the mean value for individual bar chart
        for bar, pair_data in zip(bars, top_3):
            mean_asymm = pair_data['sum'] / pair_data['freq']
            axs[idx].text(bar.get_width() *0.5, bar.get_y() + bar.get_height()/2, f'{mean_asymm:.2f}', va='center', fontsize=9)
    
    fig.suptitle(f'Frequency of highest asymmetry channel for L{L}_E{E}_tau{tau}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir+ f'/Overall-asymmetry-channel-freqs.png')
    plt.close()
    
# Correlation between subjects' age and causal asymmetry
def correl_subjects(asymm_idx_subjects, subjects_props, output_dir, L, E, tau):
    
    # Load the asymmetry index from the list of dictionaries
    c = np.array([dict['control'] for dict in asymm_idx_subjects])
    pre = np.array([dict['preictal'] for dict in asymm_idx_subjects])
    ic = np.array([dict['ictal'] for dict in asymm_idx_subjects])
    tot = np.array([c, pre, ic]).flatten()
    mx = [tot, c, pre, ic]
    
    # Pearson correlation between properties of subjects' and asymmetry index values
    correl_c, _ = pearsonr(subjects_props, c)
    correl_pre, _ = pearsonr(subjects_props, pre)
    correl_ic, _ = pearsonr(subjects_props, ic)
    correl_tot, _ = pearsonr(np.repeat(subjects_props,3), tot)

    titles = [f'Total Correl {correl_tot:.2f}', f'Control Correl {correl_c:.2f}', f'Preictal Correl {correl_pre:.2f}', f'Ictal Correl {correl_ic:.2f}']
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Correlation between age and asymmetry', fontsize=16)
    
    # Display the figures and points in respective state
    for i, (state, title) in enumerate(zip(mx, titles)):
        row = i // 2
        col = i % 2
        rep = len(state) // len(subjects_props)
        axs[row, col].scatter(np.repeat(subjects_props, rep), state)
        axs[row, col].set_title(title)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir+ "/Correlation-between-age-and-asymmetry.png")
    plt.close()    

# CCM Parameters
np.random.seed(1)
L=10000      # length of time period
tau=1       # time lag
E=2         # embedding dimensions

# Select chunk
start_index = 0
end_index = start_index + L

# Evaluate all subjects
def eval_subjects():
    
    # Lists of asymmetry indices and dominant channels
    asymm_idx_subjects = []
    asymm_ch_highest = []
    asymm_ch_highest_sub = []
    
    for subject in config.SELECTED_SUBJECTS:
        
        # Output paths
        output_dir_subj = config.OUTPUT_DIR + '/' + subject
        os.makedirs(output_dir_subj, exist_ok=True)
        output_filename_c=output_dir_subj+"/control-file"
        output_filename_ic = output_dir_subj + '/patient-ictal-file'
        output_filename_pre = output_dir_subj + '/patient-pre-ictal-file'
        
        # Plot heatmaps of each causality matrix corresponding to each state
        # plot_heatmaps(output_dir_subj, opt_L, opt_E, opt_tau, [output_filename_c, output_filename_pre, output_filename_ic], [i for i in range(1, 24)], "causality-heatmaps")
        
        # Compute asymmetry index for each subject
        asymm_idx_subjects.append(compute_asymm_idx(subject, [output_filename_c, output_filename_pre, output_filename_ic], config.OPT_L, config.OPT_E, config.OPT_TAU))
        
        # Identify only the highest asymmetric channel in each state
        asymm_ch_highest.append(compute_ch_asymm(subject, [output_filename_c, output_filename_pre, output_filename_ic], config.OPT_L, config.OPT_E, config.OPT_TAU, 1, True))
        
        # Identify the top N highest asymmetric channel in each state
        asymm_ch_highest_sub.append(compute_ch_asymm(subject, [output_filename_c, output_filename_pre, output_filename_ic], config.OPT_L, config.OPT_E, config.OPT_TAU, 1, False))
    
    # Table of the asymmetry indices for all subjects
    df = pd.DataFrame(asymm_idx_subjects)
    df.to_excel(f'{config.EVAL_DIR}/Overall-asymmetry-index.xlsx')
    
    # Table of the dominant asymmetric channels for all subjects
    df2 = pd.DataFrame(asymm_ch_highest)
    df2.to_excel(f'{config.EVAL_DIR}/Overall-asymmetry-channels.xlsx')
    
    # Plot distribution of asymmetry indices across all subjects
    plot_boxplot_asymm(asymm_idx_subjects, config.EVAL_DIR, config.OPT_L, config.OPT_E, config.OPT_TAU)
    
    # Plot frequency of dominant asymmetric channels across all subjects
    plot_frequency_asymm_ch(asymm_ch_highest_sub, config.EVAL_DIR, config.OPT_L, config.OPT_E, config.OPT_TAU)
    
    # Plot correlation between asymmetry and age of subjects
    correl_subjects(asymm_idx_subjects, np.array([11, 11, 14, 22, 7, 1.5, 3.5, 10, 3, 2, 3, 9, 16, 7, 12, 6, 13, 6]), config.EVAL_DIR, config.OPT_L, config.OPT_E, config.OPT_TAU)

