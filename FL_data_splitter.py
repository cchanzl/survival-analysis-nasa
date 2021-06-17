import pandas as pd
import numpy as np
from baseline import train_clipped, test_clipped, minmax_scaler, remaining_sensors


##########################
#   Helper Functions
##########################

def save_data_file(df, filename):
    file_path = "FL_data/"
    filename = file_path + filename + ".csv"
    df.to_csv(filename, index=False)

def fl_data_splitter(client_no, train, test):

##########################################
#   Data Preparation for FL Training
##########################################

# This section is to prepare data for use in FATE Federated Learning package
# Data used in FL must have the following headers
# id   y   x0   x1   x2   x3   ...   xN
# y must be the target label, in this case the RUL
# See examples at https://github.com/FederatedAI/FATE/tree/master/examples/data

no_of_parties = 3  # number of parties in the FL training
minmax_scale = True

if minmax_scale:
    minmax_scaler(train_clipped, test_clipped, remaining_sensors)

fl_data_splitter(no_of_parties, train_clipped, test_clipped)
