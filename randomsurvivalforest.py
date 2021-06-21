import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import pickle

################################
#   Random Survival Forest
################################


def train_rsf(rsf_x, rsf_y, rsf_filename, tune='False'):
    print("Tuning Random Survival Forest")

    # Training RSF
    random_state = 20
    if tune:  # tuned
        rsf = RandomSurvivalForest(n_estimators=775,
                                   min_samples_split=5,
                                   min_samples_leaf=5,
                                   max_features="sqrt",
                                   max_depth=671,
                                   n_jobs=-1,
                                   random_state=random_state)
    else:  # untuned
        rsf = RandomSurvivalForest(n_estimators=1000,
                                   min_samples_split=10,
                                   min_samples_leaf=15,
                                   max_features="sqrt",
                                   max_depth=10,
                                   n_jobs=-1,
                                   random_state=random_state)

    print("Fitting rsf")
    rsf.fit(rsf_x, Surv.from_dataframe('breakdown', 'cycle', rsf_y))  # y only takes structured data

    # save trained model
    pickle.dump(rsf, open(rsf_filename, 'wb'))
