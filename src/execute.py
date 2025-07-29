# Author @ Tenzin Sangpo Choedon

# General imports
import traceback
import sys

# Own imports
from config import config
from preprocess_data import *
from process_CCM_subjects import *
from eval_CCM_subjects import *
from test_surrogate_data import *
from classify_CCM_Graphwise_new import *

def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        info = traceback.extract_tb(sys.exc_info()[2])[-1]
        filename = info.filename
        line = info.lineno
        print(f"Error: {e}, File: {filename}, Line: {line})")
        return None

def execute():
    config.create_directories()
    # safe_execute(preprocess)
    # safe_execute(process_CCM)
    # safe_execute(eval_subjects)
    # safe_execute(test_surrogates_subjects)
    safe_execute(classify_states)

if __name__ == "__main__":
    execute()
