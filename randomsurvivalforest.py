import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from baseline import rsf_x_train_dropped, rsf_y_train_dropped, rsf_filename
from sksurv.util import Surv
import pickle

################################
#   Random Survival Forest
################################

print("Tuning Random Survival Forest")

# Training RSF
random_state = 20
rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=random_state)
print("Fitting rsf")
rsf.fit(rsf_x_train_dropped,
        Surv.from_dataframe('breakdown', 'cycle', rsf_y_train_dropped))  # y only takes structured data

# save trained model
pickle.dump(rsf, open(rsf_filename, 'wb'))
