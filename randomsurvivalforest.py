import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import pickle

################################
#   Random Survival Forest
################################

def train_rsf(rsf_x_train, rsf_y_train, rsf_filename):
    print("Tuning Random Survival Forest")

    # adjust data
    attribute_dropped = ['unit num', 'cycle', 'RUL', 'breakdown', 'start']  # there is no 'RUL' and 'unit num' for x_test
    rsf_x_dropped = rsf_x_train.drop(attribute_dropped, axis=1)
    rsf_y_dropped = rsf_y_train.drop('RUL', axis=1)

    # Training RSF
    random_state = 20
    rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    print("Fitting rsf")
    rsf.fit(rsf_x_dropped,
            Surv.from_dataframe('breakdown', 'cycle', rsf_y_dropped))  # y only takes structured data

    # save trained model
    pickle.dump(rsf, open(rsf_filename, 'wb'))
