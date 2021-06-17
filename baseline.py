import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress info, warning and error tensorflow messages
from datetime import datetime  # to timestamp results of each model

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
import random
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxTimeVaryingFitter
from lifelines.utils import restricted_mean_survival_time
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from sksurv.metrics import concordance_index_censored as ci_scikit
from sklearn.model_selection import RandomizedSearchCV
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
import torch
import torchtuples as tt

##########################
#   Loading Data
##########################

delimiter = "*" * 40
header = ["unit num", "cycle", "op1", "op2", "op3"]
for i in range(0, 26):
    name = "sens"
    name = name + str(i + 1)
    header.append(name)

full_path = "Dataset/"
y_test = pd.read_csv(full_path + 'RUL_FD001.txt', delimiter=" ", header=None)
df_train_001 = pd.read_csv(full_path + 'train_FD001.txt', delimiter=" ", names=header)
x_test_org = pd.read_csv(full_path + 'test_FD001.txt', delimiter=" ", names=header)

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


def evaluate(model, df_result, label='test'):
    """ Evaluates model output on rmse, R2 and C-Index
    Args:
    model (string): name of model for documentation
    df_result (pandas.df): dataframe with the headers 'unit num', 'RUL', 'y_hat', 'breakdown'
    label (string): type of output (train or test)

    Returns:
    list: returns [model, label, rmse, ci_sk, variance]
    """

    y_true = df_result['RUL']
    y_hat = df_result['y_hat']

    df_result['breakdown'].replace(0, False, inplace=True)  # rsf only takes true or false
    df_result['breakdown'].replace(1, True, inplace=True)  # rsf only takes true or false

    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)

    # the concordance index (CI) is interested on the order of the predictions, not the predictions themselves
    # CI can only be measured between individual samples where a censoring or failure event occurred
    # https://medium.com/analytics-vidhya/concordance-index-72298c11eac7#:~:text=The%20concordance%20index%20or%20c,this%20definition%20mean%20in%20practice
    df_result_grouped = df_result.groupby('unit num').last()
    breakdown = df_result_grouped['breakdown']
    y_true = df_result_grouped['RUL']
    y_hat = df_result_grouped['y_hat']
    ci_sk = ci_scikit(breakdown, y_true, y_hat)[0]
    # print(f'Number of concordant pairs (scikit-survival): {ci_scikit(breakdown, y_true, y_hat)[1]}')
    # print(f'Number of discordant pairs (scikit-survival): {ci_scikit(breakdown, y_true, y_hat)[2]}')
    # print(f'Number of pairs having tied estimated risks (scikit-survival): {ci_scikit(breakdown, y_true, y_hat)[3]}')
    # print(f'Number of comparable pairs sharing the same time (scikit-survival): {ci_scikit(breakdown, y_true, y_hat)[4]}')

    print('{} set RMSE:{:.2f}, CI(scikit):{:.4f}, R2:{:.2f}'.format(label, rmse, ci_sk, variance))
    result = [model, label, rmse, ci_sk, variance]
    return result


def exponential_model(z, a, b):
    return a * np.exp(-b * z)


def train_val_group_split(X, y, gss, groups, print_groups=False):
    for idx_train, idx_val in gss.split(X, y, groups=groups):
        X_train_split = X.iloc[idx_train].copy()
        y_train_split = y.iloc[idx_train].copy()
        X_val_split = X.iloc[idx_val].copy()
        y_val_split = y.iloc[idx_val].copy()
    return X_train_split, y_train_split, X_val_split, y_val_split


def plot_loss(fit_history):
    plt.figure(figsize=(13, 5))
    plt.plot(range(1, len(fit_history.history['loss']) + 1), fit_history.history['loss'], label='train')
    plt.plot(range(1, len(fit_history.history['val_loss']) + 1), fit_history.history['val_loss'], label='validate')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def create_model(input_dim, nodes_per_layer, dropout, activation, weights_file):
    model = Sequential()
    model.add(Dense(nodes_per_layer[0], input_dim=input_dim, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(nodes_per_layer[1], activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(nodes_per_layer[2], activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.save_weights(weights_file)
    return model


def minmax_scaler(df_train, df_test, sensor_names):
    scaler = MinMaxScaler()
    scaler.fit(df_train[sensor_names])

    # scale train set
    df_train_scaled = df_train.copy()
    df_train_scaled[sensor_names] = pd.DataFrame(scaler.transform(df_train[sensor_names]),
                                                 columns=sensor_names, index=df_train_scaled.index)
    # scale test set
    df_test_scaled = df_test.copy()
    df_test_scaled[sensor_names] = pd.DataFrame(scaler.transform(df_test[sensor_names]),
                                                columns=sensor_names,
                                                index=df_test_scaled.index)
    return df_train_scaled, df_test_scaled


def add_specific_lags(df_input, list_of_lags, columns):
    df = df_input.copy()
    for i in list_of_lags:
        lagged_columns = [col + '_lag_{}'.format(i) for col in columns]
        df[lagged_columns] = df.groupby('unit num')[columns].shift(i)
    df.dropna(inplace=True)
    return df


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()

    # first, calculate the exponential weighted mean of desired sensors
    df[sensors] = df.groupby('unit num')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean())

    # second, drop first n_samples of each unit num to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = df.groupby('unit num')['unit num'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]

    return df


# prepare data for neural network
def prep_data(x_train, y_train, x_test, remaining_sensors, lags, alpha, n=0):
    X_train_interim, X_test_interim = minmax_scaler(x_train, x_test, remaining_sensors)

    X_train_interim = exponential_smoothing(X_train_interim, remaining_sensors, n, alpha)
    X_test_interim = exponential_smoothing(X_test_interim, remaining_sensors, n, alpha)

    # X_train_interim = add_specific_lags(X_train_interim, lags, remaining_sensors)
    # X_test_interim = add_specific_lags(X_test_interim, lags, remaining_sensors)

    # train_idx = X_train_interim.index
    # y_train = y_train.iloc[train_idx]

    return X_train_interim, y_train, X_test_interim


# Estimate remaining useful life
def evaluate_rsf(name, model, rsf_x, label):
    surv = model.predict_survival_function(rsf_x[rsf_predict_cols], return_array=True)  # return S(t) for each data point

    rmst = []  # create a list to store the rmst of each survival curve
    # calculate rmst

    for i, s in enumerate(surv):
        df_s = pd.DataFrame(s)
        df_s.set_index(model.event_times_, inplace=True)
        km_rmst = restricted_mean_survival_time(df_s, t=400)  # calculate restricted mean survival time
        rmst.append(km_rmst)
        # np.set_printoptions(threshold=np.inf)
        # if (i % 9) == 0:
        #    print(km_rmst)
        # plt.step(rsf.event_times_, s, where="post", label=str(i))

    # plt.ylabel("Survival probability")
    # plt.xlabel("Time in cycles")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    rsf_x['y_hat'] = rmst - rsf_x['cycle']
    rsf_x['y_hat'].where(rsf_x['y_hat'] >= 0, 0, inplace=True)
    rsf_x['y_hat'] = rsf_x['y_hat'].clip(upper=clip_level)
    result_interim = evaluate(name, rsf_x, label)
    return result_interim, rsf_x['y_hat']


################################
#   Global variables
################################

# set upper bound to integrate survival function to find RMST
# kaplan-meier and rsf is highly sensitive to this as they use this to integrate S(t)
rmst_upper_bound = 200

# clip RUL if above a certain level to improve training
clip_level = 150

# set if pseudo right censoring be performed on dataset at designated cut-off
right_censoring = True
cut_off = 200

# to re-train untuned baseline NN?
train_untuned_NN = False

# to perform hyperparameter search for NN?
nn_hyperparameter_tune = False
nn_n_Iterations = 500
train_tuned_NN = False

# to re-train untuned rsf
train_untuned_rsf = True

# to perform hyperparameter search for rsf?
rsf_hyperparameter_tune = False
rsf_n_Iterations = 500
train_tuned_rsf = False

# If graph should be displayed at the end
show_graph = True

################################
#   General Data pre-processing
################################

print(delimiter)
print("Started data preprocessing")

# Sensors selected based on MannKendall, sensor 2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20 and 21 are selected
drop_sensors = ['sens1', 'sens5', 'sens6', 'sens9', 'sens10', 'sens14',
                'sens16', 'sens18', 'sens19', 'sens22', 'sens23', 'sens24', 'sens25', 'sens26']

drop_labels = drop_sensors + ["op1", "op2", "op3"]

remaining_sensors = ['sens2', 'sens3', 'sens4', 'sens7', 'sens8', 'sens11',
                     'sens12', 'sens13', 'sens15', 'sens17', 'sens20', 'sens21']

all_sensors = drop_sensors + remaining_sensors

# add RUL column
train_org = add_remaining_useful_life(df_train_001)
test_org = x_test_org.copy()
y_test.rename(columns={0: 'RUL'}, inplace=True)
y_test['unit num'] = [i for i in range(1, 101)]
y_test['max_cycle'] = y_test['RUL'] + test_org.groupby('unit num').last().reset_index(drop=True)['cycle']
test_org = pd.merge(test_org, y_test, on='unit num', how='left')
test_org['RUL'] = test_org['max_cycle'] - test_org['cycle']
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(test_org[['unit num', 'cycle', 'RUL', 'max_cycle']])
test_org.drop(['max_cycle'], axis=1, inplace=True)

# drop non-informative sensors and operational settings
train_org.drop(labels=drop_labels, axis=1, inplace=True)
test_org.drop(labels=drop_labels, axis=1, inplace=True)
test_org.drop([1], axis=1, inplace=True)

# add event indicator 'breakdown' column
train_org['breakdown'] = 0
idx_last_record = train_org.reset_index().groupby(by='unit num')['index'].last()  # engines breakdown at the last cycle
train_org.at[idx_last_record, 'breakdown'] = 1
test_org['breakdown'] = 0
idx_last_record = test_org.reset_index().groupby(by='unit num')['index'].last()  # engines breakdown at the last cycle
test_org.at[idx_last_record, 'breakdown'] = 1

# Add start cycle column (only required for Cox model)
train_org['start'] = train_org['cycle'] - 1
test_org['start'] = test_org['cycle'] - 1

# apply a floor to RUL in training data. 125 is selected as the min cycle is 128. See EDA.
train_clipped = train_org.copy()
train_clipped['RUL'] = train_clipped['RUL'].clip(upper=clip_level)
test_clipped = test_org.copy()
test_clipped['RUL'] = test_clipped['RUL'].clip(upper=clip_level)

# Apply psuedo right censoring in training data. Important to improve accuracy and simulate right censoring
if right_censoring:
    train_clipped = train_clipped[train_clipped['cycle'] <= cut_off].copy()

# create df to append result of each model
evaluation_metrics = ["RMSE", "CI_SK", "R2"]
results_header = ["model_name", "train_test"] + evaluation_metrics
list_results = []

# create df to append y_hat and y_pred for graph plotting
graph_data = test_clipped.copy()

################################
#   Kaplan-Meier Curve
################################

print(delimiter)
print("Started Kaplan-Meier")

# Initilise columns used for training and prediction
train_cols = ['unit num', 'cycle'] + remaining_sensors + ['start', 'breakdown']
predict_cols = ['cycle'] + remaining_sensors + ['start', 'breakdown']  # breakdown value will be 0

km_train = train_clipped[['unit num', 'cycle', 'breakdown', 'RUL']].groupby('unit num').last()

plt.figure(figsize=(15, 7))
kaplanMeier = KaplanMeierFitter()
kaplanMeier.fit(km_train['cycle'], km_train['breakdown'])

# kaplanMeier.plot()
# plt.ylabel("Probability of survival")
# plt.show()
# plt.close()

# estimate restricted mean survival time from KM curve
km_rmst = restricted_mean_survival_time(kaplanMeier, t=rmst_upper_bound)
df_result = train_clipped.copy()
df_result['km_rmst'] = km_rmst
km_rmst_arr = [km_rmst for x in range(len(df_result))]
df_result['y_hat'] = km_rmst_arr - df_result['cycle']
df_result['y_hat'].where(df_result['y_hat'] >= 0, 0, inplace=True)
# y_hat = y_hat.clip(upper=clip_level)
result = evaluate("KM_rmst", df_result, 'train')
list_results.append(result)

km_rmst_arr = [km_rmst for x in range(len(test_clipped))]
df_result = test_clipped.copy()
df_result['y_hat'] = km_rmst_arr - df_result['cycle']
df_result['y_hat'].where(df_result['y_hat'] >= 0, 0, inplace=True)
# y_hat = y_hat.clip(upper=clip_level)
result = evaluate("KM_rmst", df_result, 'test')
list_results.append(result)

graph_data['km_rmst'] = df_result['y_hat']

################################
#   Cox-PH Model
################################

print(delimiter)
print("Started Cox PH")

# Train Cox model
ctv = CoxTimeVaryingFitter()
ctv.fit(train_clipped[train_cols], id_col="unit num", event_col='breakdown',
        start_col='start', stop_col='cycle', show_progress=True, step_size=1)

# Calculate log_partial_hazard for all data points
train_cox = train_clipped.copy()  # need to make a copy so that we can add 'hazard' later
predictions = ctv.predict_log_partial_hazard(train_cox)
train_cox['hazard'] = predictions.to_frame()[0].values

# df_hazard.plot('hazard', 'RUL', 'scatter', figsize=(15,5))
# plt.xlabel('hazard')
# plt.ylabel('RUL')
# plt.show()
# plt.close()

# Fit an exponential curve to the relationship between log_partial_hazard and RUL
popt, pcov = curve_fit(exponential_model, train_cox['hazard'], train_cox['RUL'])

# perform prediction solely on log-partial hazard and evaluate
train_cox['y_hat'] = exponential_model(train_cox['hazard'], *popt)
print("fitted exponential curve")
result = evaluate("Cox", train_cox, 'train')
list_results.append(result)

y_pred = ctv.predict_log_partial_hazard(test_clipped)
df_result = test_clipped.copy()
df_result['y_hat'] = exponential_model(y_pred, *popt)
result = evaluate('Cox', df_result, 'test')
list_results.append(result)
graph_data['Cox'] = df_result['y_hat']

################################
#   Random Forest
################################

# https://towardsdatascience.com/random-forest-for-predictive-maintenance-of-turbofan-engines-5260597e7e8f

print(delimiter)
print("Started Random Forest")

# data preparation
rf_x = train_clipped.copy()
rf_y = rf_x.pop('RUL')
rf_x_train, rf_x_val, rf_y_train, rf_y_val = train_test_split(rf_x, rf_y, test_size=0.25, random_state=6)

# split scaled dataset into train test split
# gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
# rf_x_train, rf_x_val, rf_y_train, rf_y_val = train_val_group_split(rf_x, rf_y, gss, rf_x['unit num'])

# model fitting
rf = RandomForestRegressor(n_estimators=100, criterion="mse", max_features="sqrt", random_state=42)
rf.fit(rf_x_train[remaining_sensors], rf_y_train)

# predict and evaluate, without any hyperparameter tuning
rf_x_val['y_hat'] = rf.predict(rf_x_val[remaining_sensors])
rf_x_val['RUL'] = rf_y_val
print("pre-tuned RF")
result = evaluate('RF (pre-tuned)', rf_x_val, 'train')
list_results.append(result)

df_result = test_clipped.copy()
df_result['y_hat'] = rf.predict(test_clipped[remaining_sensors])
result = evaluate("RF (pre-tuned)", df_result, 'test')
list_results.append(result)
graph_data['RF (pre-tuned)'] = df_result['y_hat']

# perform some checks on layout of a SINGLE tree
# print(rf.estimators_[5].tree_.max_depth)  # check how many nodes in the longest path
# rf.estimators_[5].tree_.n_node_samples    # check how many samples in the last nodes

# crudely tweaked random forest
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42, max_depth=8, min_samples_leaf=50)
rf.fit(rf_x_train[remaining_sensors], rf_y_train)

# predict and evaluate
rf_x_val['y_hat'] = rf.predict(rf_x_val[remaining_sensors])
print("crudely tuned RF")
result = evaluate('RF (tuned)', rf_x_val, 'train')
list_results.append(result)

df_result['y_hat'] = rf.predict(test_clipped[remaining_sensors])
result = evaluate("RF (tuned)", df_result, 'test')
list_results.append(result)
graph_data['RF (tuned)'] = df_result['y_hat']

################################
#   Neural Network
################################
# https://towardsdatascience.com/lagged-mlp-for-predictive-maintenance-of-turbofan-engines-c79f02a15329

print(delimiter)
print("Started Neural Network")
# make a copy of the original full dataset as we need to scale it for NN
train_NN = train_clipped.copy()

# scaling using minmax
nn_y_train = train_NN.pop('RUL')
nn_x_train_scaled, nn_x_test_scaled = minmax_scaler(train_NN, test_clipped, remaining_sensors)
# split scaled dataset into train test split
gss = GroupShuffleSplit(n_splits=1, train_size=0.80,
                        random_state=42)  # even though we set np and tf seeds, gss requires its own seed
nn_x_train_scaled, nn_y_train, nn_x_val_scaled, nn_y_val = train_val_group_split(nn_x_train_scaled,
                                                                                 nn_y_train, gss,
                                                                                 nn_x_train_scaled['unit num'])

# training the model
filename = 'finalized_pretuned_NN_model.h5'
if train_untuned_NN:
    # construct neural network
    ct_untuned = Sequential()
    ct_untuned.add(Dense(16, input_dim=len(remaining_sensors), activation='relu'))
    ct_untuned.add(Dense(32, activation='relu'))
    ct_untuned.add(Dense(64, activation='relu'))
    ct_untuned.add(Dense(1))
    ct_untuned.compile(loss='mean_squared_error', optimizer='adam')

    epochs = 20
    history = ct_untuned.fit(nn_x_train_scaled[remaining_sensors], nn_y_train,
                             validation_data=(nn_x_val_scaled[remaining_sensors], nn_y_val),
                             epochs=epochs, verbose=0)
    ct_untuned.save(filename)  # save trained model

ct_untuned = load_model(filename)
nn_x_val_scaled['y_hat'] = ct_untuned.predict(nn_x_val_scaled[remaining_sensors])
nn_x_val_scaled['RUL'] = nn_y_val
print("pre-tuned Neural Network")
result = evaluate("NN (pre-tuned)", nn_x_val_scaled, 'train')
list_results.append(result)

# nn_x_test_scaled = nn_x_test_scaled.drop(['cycle', 'RUL', 'start'], axis=1)
df_result = test_clipped.copy()
df_result['y_hat'] = ct_untuned.predict(nn_x_test_scaled[remaining_sensors])
result = evaluate("NN (pre-tuned)", df_result, 'test')
list_results.append(result)
graph_data['NN (pre-tuned)'] = df_result['y_hat']

# Hyperparameter tuning
if nn_hyperparameter_tune:
    alpha_list = list(np.arange(5, 20 + 1, 0.5) / 100)
    epoch_list = list(np.arange(10, 50 + 1, 5))
    nodes_list = [[8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512]]

    # lowest dropout=0.1, because I know zero dropout will yield better training results but worse generalization (overfitting)
    dropouts = list(np.arange(0, 4 + 1, 0.5) / 10)

    # earlier testing revealed relu performed significantly worse, so I removed it from the options
    activation_functions = ['tanh', 'sigmoid', 'relu']
    batch_size_list = [16, 32, 64, 128, 256, 512]

    ITERATIONS = nn_n_Iterations
    results = pd.DataFrame(columns=['MSE', 'std_MSE',  # bigger std means less robust
                                    'alpha', 'epochs',
                                    'nodes', 'dropout',
                                    'activation', 'batch_size'])

    weights_file = 'mlp_hyper_parameter_weights.h5'  # save model weights
    specific_lags = [1, 2, 3, 4, 5, 10, 20]

    for i in range(ITERATIONS):
        print("Iteration ", str(i + 1))
        mse = []

        # init parameters
        alpha = random.sample(alpha_list, 1)[0]
        epochs = random.sample(epoch_list, 1)[0]
        nodes_per_layer = random.sample(nodes_list, 1)[0]
        dropout = random.sample(dropouts, 1)[0]
        activation = random.sample(activation_functions, 1)[0]
        batch_size = random.sample(batch_size_list, 1)[0]

        # create dataset
        nn_x_train, nn_y_train, _ = prep_data(x_train=train_clipped,
                                              y_train=train_clipped['RUL'],
                                              x_test=test_clipped,
                                              remaining_sensors=remaining_sensors,
                                              lags=specific_lags,
                                              alpha=alpha)
        # create model
        input_dim = len(nn_x_train[remaining_sensors].columns)
        ct_untuned = create_model(input_dim, nodes_per_layer, dropout, activation, weights_file)
        # create train-validation split
        gss_search = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)
        for idx_train, idx_val in gss_search.split(nn_x_train, nn_y_train, groups=train_clipped['unit num']):
            X_train_split = nn_x_train.iloc[idx_train].copy()
            y_train_split = nn_y_train.iloc[idx_train].copy()
            X_val_split = nn_x_train.iloc[idx_val].copy()
            y_val_split = nn_y_train.iloc[idx_val].copy()

            # train and evaluate model
            ct_untuned.compile(loss='mean_squared_error', optimizer='adam')
            ct_untuned.load_weights(weights_file)  # reset optimizer and node weights before every training iteration
            history = ct_untuned.fit(X_train_split[remaining_sensors], y_train_split,
                                     validation_data=(X_val_split[remaining_sensors], y_val_split),
                                     epochs=epochs, batch_size=batch_size, verbose=0)
            mse.append(history.history['val_loss'][-1])

        # append results
        d = {'MSE': np.mean(mse), 'std_MSE': np.std(mse), 'alpha': alpha,
             'epochs': epochs, 'nodes': str(nodes_per_layer), 'dropout': dropout,
             'activation': activation, 'batch_size': batch_size}
        results = results.append(pd.DataFrame(d, index=[0]), ignore_index=True)

    results.to_csv("nn_hyp_results_" + now.replace('/', '-').replace(' ', '_').replace(':', '') + ".csv", index=False)

alpha = 0.175
epochs = 40
specific_lags = [1, 2, 3, 4, 5, 10, 20]
nodes = [128, 256, 512]
dropout = 0.05
activation = 'sigmoid'
batch_size = 16

nn_x_train, nn_y_train, nn_x_test = prep_data(x_train=train_clipped,
                                              y_train=train_clipped['RUL'],
                                              x_test=test_clipped,
                                              remaining_sensors=remaining_sensors,
                                              lags=specific_lags,
                                              alpha=alpha)
filename = 'finalized_tuned_NN_model.h5'
if train_tuned_NN:
    input_dim = len(nn_x_train[remaining_sensors].columns)
    weights_file = 'mlp_hyper_parameter_weights'
    nn_lagged_tuned = create_model(input_dim,
                                   nodes_per_layer=nodes,
                                   dropout=dropout,
                                   activation=activation,
                                   weights_file=weights_file)

    nn_lagged_tuned.compile(loss='mean_squared_error', optimizer='adam')
    nn_lagged_tuned.load_weights(weights_file)
    nn_lagged_tuned.fit(nn_x_train[remaining_sensors], nn_y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    nn_lagged_tuned.save(filename)  # save trained model

# predict and evaluate
nn_lagged_tuned = load_model(filename)
print("tuned Neural Network")
nn_x_train['y_hat'] = nn_lagged_tuned.predict(nn_x_train[remaining_sensors])
nn_x_train['RUL'] = nn_y_train
result = evaluate("NN (tuned)", nn_x_train, 'train')
list_results.append(result)

df_result = test_clipped.copy()
df_result['y_hat'] = nn_lagged_tuned.predict(nn_x_test[remaining_sensors])
result = evaluate("NN (tuned)", df_result, 'test')
list_results.append(result)
graph_data['NN (tuned)'] = df_result['y_hat']

################################
#   Random Survival Forest
################################

print(delimiter)
print("Started Random Survival Forest")

# Data preparation
# rsf_x = train_clipped.copy()  # cannot use clip as we need the last RUL of the engine to train the model
rsf_x = train_org.copy()  # hence we are using train_org
rsf_x = rsf_x.reset_index().groupby(by='unit num').last()
breakdown_0 = train_org.copy()
breakdown_0 = breakdown_0.loc[breakdown_0['breakdown'] == 0]
breakdown_0 = breakdown_0.sample(n=300)
rsf_x.append(breakdown_0, ignore_index=True)  # mix of breakdown and non-breakdown
rsf_x['RUL'] = rsf_x['RUL'].clip(upper=clip_level)
rsf_x['RUL'] = rsf_x.RUL.astype('float')

rsf_y = rsf_x[['breakdown', 'cycle']]
rsf_y['breakdown'].replace(0, False, inplace=True)  # rsf only takes true or false
rsf_y['breakdown'].replace(1, True, inplace=True)  # rsf only takes true or false
rsf_x_train, rsf_x_val, rsf_y_train, rsf_y_val = train_test_split(rsf_x, rsf_y, test_size=0.25, random_state=7)

# Training RSF
print("Predicting rsf")
rsf_filename = 'finalized_untuned_rsf_model_incl_cycle.sav'
rsf_predict_cols = remaining_sensors.copy()
# rsf_predict_cols.append('cycle')
if train_untuned_rsf:
    from randomsurvivalforest import train_rsf
    train_rsf(rsf_x_train[rsf_predict_cols], rsf_y_train, rsf_filename)
rsf = pickle.load(open(rsf_filename, 'rb'))

result, _ = evaluate_rsf("rsf (pre-tuned)", rsf, rsf_x_val, 'train')
list_results.append(result)

result, y_hat = evaluate_rsf("rsf (pre-tuned)", rsf, test_clipped, 'test')
list_results.append(result)
graph_data['rsf (pre-tuned)'] = y_hat

# Hyperparameter tuning RSF
if rsf_hyperparameter_tune:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=150, num=5)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 120, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Minimum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(start=5, stop=30, num=1)]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(start=5, stop=30, num=1)]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    # print(random_grid)

    # perform random search
    rsf = RandomSurvivalForest()
    rsf_random = RandomizedSearchCV(estimator=rsf,
                                    param_distributions=random_grid,
                                    n_iter=rsf_n_Iterations,
                                    cv=5,
                                    verbose=10,
                                    random_state=5,
                                    n_jobs=-1)
    # Fit the random search model
    rsf_random.fit(rsf_x[remaining_sensors], Surv.from_dataframe('breakdown', 'cycle', rsf_y))
    print(rsf_random.best_params_)

print("done with tuning, start training")
rsf_filename = 'finalized_tuned_rsf_model.sav'
if train_tuned_rsf:
    from randomsurvivalforest import train_rsf
    train_rsf(rsf_x[rsf_predict_cols], rsf_y, rsf_filename, True)  # True to train tuned model
rsf_tuned = pickle.load(open(rsf_filename, 'rb'))

result, _ = evaluate_rsf("rsf (tuned)", rsf_tuned, rsf_x_val, 'train')
list_results.append(result)

result, y_hat = evaluate_rsf("rsf (tuned)", rsf_tuned, test_clipped, 'test')
list_results.append(result)
graph_data['rsf (tuned)'] = y_hat

################################
#   Cox-Time Method
################################

# https://jmlr.org/papers/volume20/18-424/18-424.pdf
# https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html

print(delimiter)
print("Started CoxTime Method")

np.random.seed(1234)
_ = torch.manual_seed(123)

# Prepare data
ct_x = rsf_x.copy()
ct_y = rsf_y.copy()

ct_x_scaled, ct_test_scaled = minmax_scaler(ct_x, test_clipped, remaining_sensors)

ct_x_train, ct_x_val, ct_y_train, ct_y_val = train_test_split(ct_x_scaled, ct_y, test_size=0.25, random_state=7)

labtrans = CoxTime.label_transform()
get_target = lambda df: (df['cycle'].values.astype('float32'), df['breakdown'].values.astype('float32'))
ct_y_train_split_tuple = labtrans.fit_transform(*get_target(ct_y_train))
ct_y_val_split_tuple = labtrans.transform(*get_target(ct_y_val))
val = tt.tuplefy(ct_x_val[remaining_sensors].to_numpy().astype('float32'), ct_y_val_split_tuple)

# Prepare model
in_features = ct_x_train[remaining_sensors].shape[1]
num_nodes = [32, 32]
batch_norm = True
dropout = 0.1
batch_size = 256

net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
ct_untuned = CoxTime(net, tt.optim.Adam, labtrans=labtrans)  # set labtrans to get back the correct time scale in output

lrfinder = ct_untuned.lr_finder(ct_x_train[remaining_sensors].to_numpy().astype('float32'),
                                ct_y_train_split_tuple, batch_size, tolerance=2)
_ = lrfinder.plot()
# plt.show()

print("The best learning rate is: ", str(lrfinder.get_best_lr()))

ct_untuned.optimizer.set_lr(lrfinder.get_best_lr())

log = ct_untuned.fit(ct_x_train[remaining_sensors].to_numpy().astype('float32'), ct_y_train_split_tuple,
                     batch_size,
                     epochs=512,
                     callbacks=[tt.callbacks.EarlyStopping()],
                     verbose=True,
                     val_data=val.repeat(10).cat())
_ = log.plot()
plt.show()

print(ct_untuned.partial_log_likelihood(*val).mean())
_ = ct_untuned.compute_baseline_hazards()
surv = ct_untuned.predict_surv_df(ct_x_val[remaining_sensors].to_numpy().astype('float32'))

# print(test_clipped.shape)
# print(surv)
surv.iloc[:, :20].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
plt.show()

ct_rmst = []  # create a list to store the rmst of each survival curve
for col in surv:
    km_rmst = restricted_mean_survival_time(surv[col].to_frame(), t=400)  # calculate restricted mean survival time
    ct_rmst.append(km_rmst)

ct_x_val['y_hat'] = ct_rmst - ct_x_val['cycle']
ct_x_val['y_hat'].where(ct_x_val['y_hat'] >= 0, 0, inplace=True)
ct_x_val['y_hat'] = ct_x_val['y_hat'].clip(upper=clip_level)
result = evaluate("ct (untuned)", ct_x_val, 'train')
list_results.append(result)

surv = ct_untuned.predict_surv_df(ct_test_scaled[remaining_sensors].to_numpy().astype('float32'))

ct_rmst = []  # create a list to store the rmst of each survival curve
for col in surv:
    km_rmst = restricted_mean_survival_time(surv[col].to_frame(), t=400)  # calculate restricted mean survival time
    ct_rmst.append(km_rmst)

ct_test_scaled['y_hat'] = ct_rmst - ct_test_scaled['cycle']
ct_test_scaled['y_hat'].where(ct_x_val['y_hat'] >= 0, 0, inplace=True)
ct_test_scaled['y_hat'] = ct_test_scaled['y_hat'].clip(upper=clip_level)
result = evaluate("ct (untuned)", ct_test_scaled, 'test')
list_results.append(result)
graph_data["ct (untuned)"] = y_hat

################################
#   Save results of each model
################################

print(delimiter)
print("Saving all results")
df_results = pd.DataFrame(list_results, columns=results_header)
df_results["timestamp"] = now

# format numbers
for col in evaluation_metrics:
    df_results[col] = pd.Series(["{0:.4f}".format(val) for val in df_results[col]], index=df_results.index)

# sort values
df_results.sort_values(by=['train_test', 'RMSE'], ascending=[True, True], inplace=True)

# print to console and save
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_results)

# save file
full_path = "results/"
filename = full_path + "saved_results_" + now.replace('/', '-').replace(' ', '_').replace(':', '') + ".csv"
df_results.to_csv(filename, index=False, sep='\t')
graph_data.to_csv("graphing.csv", index=False)

# show graph
if show_graph:
    from graphing import make_graph

    make_graph([31, 38, 78, 91, 5, 46, 55, 82])
