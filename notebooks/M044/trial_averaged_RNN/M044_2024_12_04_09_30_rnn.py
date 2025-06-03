# general imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pyaldata package
import sys
import os
import importlib
sys.path.append("/home/zms24/Desktop")
import PyalData.pyaldata as pyal

# my own functions for RNN and CURBD running
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from tools.curbd import curbd
from tools.dsp.preprocessing import preprocess
from tools.rnn_and_curbd import rnn as rnnz
from tools.rnn_and_curbd import plotting as pltz
from tools.rnn_and_curbd import model_analysis as analyz
from tools.rnn_and_curbd import curbd as curbdz


importlib.reload(rnnz)
importlib.reload(pltz)
importlib.reload(analyz)
importlib.reload(curbdz)

np.random.seed(44)

### IMPORT DATA ###
