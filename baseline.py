import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

##########################
#   Loading Data
##########################

header = ["unit num", "cycle", "op1", "op2", "op3"]
for i in range(0, 26):
    name = "sens"
    name = name + str(i + 1)
    header.append(name)

full_path = "C:/Users/chanzl_thinkpad/Dropbox/Imperial/Individual Project/NASA/Dataset/"
y_test = pd.read_csv(full_path + 'RUL_FD001.txt', delimiter=" ", header=None)
df_train_001 = pd.read_csv(full_path + 'train_FD001.txt', delimiter=" ", names=header)
x_test = pd.read_csv(full_path + 'test_FD001.txt', delimiter=" ", names=header)


##########################
#   Helper Functions
##########################

def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit num")
    max_cycle = grouped_by_unit["cycle"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit num', right_index=True)

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["cycle"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


################################
#   General Data preprocessing
################################

# add RUL column
train = add_remaining_useful_life(df_train_001)

# apply a floor to RUL
# 125 is selected as the min cycle is 128. See EDA.
train['RUL'].clip(upper=125, inplace=True)

# Based on MannKendall, sensor 2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20 and 21 are selected
drop_labels = ["op1", "op2", "op3", 'sens1', 'sens5', 'sens6', 'sens9', 'sens10', 'sens14', 'sens16', 'sens18', 'sens19'
                , 'sens22', 'sens23', 'sens24', 'sens25', 'sens26']

remaining_sensors = ['sens2', 'sens3', 'sens4', 'sens7', 'sens8', 'sens11',
                     'sens12', 'sens13', 'sens15', 'sens17', 'sens20', 'sens21']

# drop_labels = ["op1", "op2", "op3",'sens1','sens5','sens6','sens10','sens16','sens18','sens19','sens22','sens23',
# 'sens24','sens25','sens26']

# remaining_sensors = ['sens2', 'sens3', 'sens4', 'sens7', 'sens8', 'sens9', 'sens11', 'sens12', 'sens13', 'sens14',
#                   'sens15', 'sens17', 'sens20', 'sens21']

train.drop(labels=drop_labels, axis=1, inplace=True)

# Add event indicator column
train['breakdown'] = 0
idx_last_record = train.reset_index().groupby(by='unit num')['index'].last()  # engines breakdown at the last cycle
train.at[idx_last_record, 'breakdown'] = 1

# Add start cycle column
train['start'] = train['cycle'] - 1

# Apply cut-off
cut_off = 200 # Important to improve accuracy
train_censored = train[train['cycle'] <= cut_off].copy() # Final dataset to use for all model

################################
#   Kaplan-Meier Curve
################################

from lifelines import KaplanMeierFitter, CoxTimeVaryingFitter

data = train_censored[['unit num', 'cycle','breakdown']].groupby('unit num').last()

plt.figure(figsize=(15,7))
survival = KaplanMeierFitter()
survival.fit(data['cycle'], data['breakdown'])
survival.plot()
plt.ylabel("Probability of survival")
plt.show()
plt.close()

################################
#   Cox-PH Model
################################