# Author @Tenzin Sangpo Choedon

import os

class Config:
    def __init__(self):
        
        # Paths
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PARENT_DIR = os.path.dirname(self.BASE_DIR)
        self.DATA_DIR = os.path.join(self.PARENT_DIR, os.path.join('data', "chbmit-1.0.0.physionet.org"))
        self.OUTPUT_DIR = os.path.join(self.PARENT_DIR, 'output_data')
        self.PROC_DIR = os.path.join(self.PARENT_DIR, 'processed_data')
        self.GRAPHS_DIR = os.path.join(self.PARENT_DIR, 'graphs_data')
        self.EVAL_DIR = os.path.join(self.PARENT_DIR, 'eval_data')
        
        # Subjects
        self.SELECTED_SUBJECTS = [f"chb{str(i).zfill(2)}" for i in [1, 2, 3,4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 20, 21, 23]]
        self.ALL_SUBJECTS = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]
        self.TEST_SUBJECT = "chb01"
        
        # CCM ranges
        self.L_RANGE = [6000, 7000, 8000, 9000, 10000]
        self.E_RANGE = [2, 3, 4, 5]
        self.TAU_RANGE=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Optimal CCM parameters (Observed from analysis) 
        self.OPT_L = 10000
        self.OPT_TAU = 4
        self.OPT_E = 4
        
        # Channels
        self.LIMIT_CHANNELS = [2, 4, 6, 7]                          # Limit channels for CCM parameter testing
        self.ALL_CHANNELS = list(range(1, 24))
        self.NUM_CHANNELS = 23
        
        # Dominant asymmetric channels (Observed from analysis)
        self.REDUCE_CHANNELS = {
            0: [5, 21, 22],
            1: [5, 18, 20],
            2: [5, 6, 12]
        }
        
        # Classes and labels
        self.STATES = {
            0: "Non-seizure",
            1: "Pre-seizure",
            2: "Seizure"
        }
        self.NUM_STATES = len(self.STATES)
        self.LABELS = list(self.STATES.keys())
        
    def create_directories(self):
        os.makedirs(self.PROC_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.GRAPHS_DIR, exist_ok=True)
        os.makedirs(self.EVAL_DIR, exist_ok=True)

config = Config()
