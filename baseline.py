import os

from tensorflow.python.keras.utils.np_utils import to_categorical

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
from data_adjustment import minmax_scaler, clip_level, remaining_sensors

##########################
#   Loading Data
##########################

delimiter = "*" * 40
header = ["unit num", "cycle", "op1", "op2", "op3"]
for i in range(0, 26):
    name = "sens"
    name = name + str(i + 1)
    header.append(name)

full_path = "Dataset/processed/"
train_org = pd.read_csv(full_path + 'train_org.csv')
test_org = pd.read_csv(full_path + 'test_org.csv')
train_clipped = pd.read_csv(full_path + 'train_clipped.csv')
test_clipped = pd.read_csv(full_path + 'test_clipped.csv')
train_trend = pd.read_csv(full_path + 'rul_rf_train_trended.csv')
test_trend = pd.read_csv(full_path + 'rul_rf_test_trended.csv')
test_trend_class = pd.read_csv(full_path + 'rul_rf_test_trended_classified.csv')
train_trend_class = pd.read_csv(full_path + 'rul_rf_train_trended_classified.csv')

# create df to append y_hat and y_pred to store complete prediction for test and train
df_master_test = test_clipped.copy()
df_master_train = train_clipped.copy()

# create df to append result of each model
evaluation_metrics = ["RMSE", "Score", "CI_SK", "R2"]
results_header = ["model_name", "train_test"] + evaluation_metrics
list_results = []


##########################
#   Helper Functions
##########################


def nasaScore(RUL_true, RUL_hat):
    d = RUL_hat - RUL_true
    score = 0
    for i in d:
        if i >= 0:
            score += np.math.exp(i / 13) - 1
        else:
            score += np.math.exp(- i / 10) - 1
    return score / len(RUL_true)  # should the score be averaged?


def evaluate(model, df_result, label='test'):
    """ Evaluates model output on rmse, R2 and C-Index
    Args:
    model (string): name of model for documentation
    df_result (pandas.df): dataframe with the headers 'unit num', 'RUL', 'y_hat', 'breakdown'
    label (string): type of output (train or test)

    Returns:
    list: returns [model, label, rmse, score, ci_sk, variance]
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
    score = nasaScore(y_true, y_hat)  # score should be based on the last instance
    # print(f'Number of concordant pairs (scikit-survival): {ci_scikit(breakdown, y_true, y_hat)[1]}')
    # print(f'Number of discordant pairs (scikit-survival): {ci_scikit(breakdown, y_true, y_hat)[2]}')
    # print(f'Number of pairs having tied estimated risks (scikit-survival): {ci_scikit(breakdown, y_true, y_hat)[3]}')
    # print(f'Number of comparable pairs sharing the same time (scikit-survival): {ci_scikit(breakdown, y_true, y_hat)[4]}')
    print('{} set RMSE:{:.2f}, Score:{:.2f}, CI(scikit):{:.4f}, R2:{:.2f}'.format(label, rmse, score, ci_sk, variance))
    result = [model, label, rmse, score, ci_sk, variance]
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


def create_model(input_dim, nodes_per_layer, dropout, activation, weights_file, nb_classes=16, type='regression'):
    model = Sequential()
    model.add(Dense(nodes_per_layer[0], input_dim=input_dim, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(nodes_per_layer[1], activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(nodes_per_layer[2], activation=activation))
    model.add(Dropout(dropout))
    if type == 'classification':
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
    else:
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

    model.save_weights(weights_file)
    return model


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
def evaluate_rsf(name, model, x_data, label, rmst_upper=400):
    surv = model.predict_survival_function(x_data[rsf_predict_cols],
                                           return_array=True)  # return S(t) for each data point
    if label == "test":
        x_data_ = x_data.reset_index().groupby(by='unit num').last()
        surv = model.predict_survival_function(x_data_[rsf_predict_cols],
                                               return_array=True)  # return S(t) for each data point

    rmst = []  # create a list to store the rmst of each survival curve
    # calculate rmst

    for i, s in enumerate(surv):
        df_s = pd.DataFrame(s)
        df_s.set_index(model.event_times_, inplace=True)
        model_RMST = restricted_mean_survival_time(df_s, t=rmst_upper)  # calculate restricted mean survival time
        rmst.append(model_RMST)
    #   plt.step(rsf.event_times_, s, where="post", label=str(i))
    #
    # plt.ylabel("Survival probability")
    # plt.xlabel("Time in cycles")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    if label == "test":
        x_data['y_hat'] = -1
        for unit in range(0, x_data['unit num'].nunique()):
            mask = (x_data['unit num'] == unit + 1)
            x_data.loc[mask, 'y_hat'] = rmst[unit] - x_data['cycle']

    else:
        x_data['y_hat'] = rmst - x_data['cycle']

    # x_data['y_hat'].where(x_data['y_hat'] >= 0, 0, inplace=True)
    x_data['y_hat'] = x_data['y_hat'].clip(upper=clip_level)
    result_interim = evaluate(name, x_data, label)
    return result_interim, x_data['y_hat']


def map_test_result(df_temp, df_main, L=20):
    interim_result = df_main.copy()
    interim_result['y_hat'] = 0
    count = 0
    for engine in interim_result['unit num'].unique():
        # get first and last index position of each set of engine
        first_idx = interim_result['unit num'].eq(engine).idxmax()
        last_idx = interim_result['unit num'].eq(engine + 1).idxmax() - 1
        if last_idx == -1:
            last_idx = len(interim_result) - 1

        # populate RUL for middle cycles
        while (last_idx - first_idx) >= L - 1:
            mid_idx = first_idx + L - 1
            interim_result.iat[mid_idx, -1] = df_temp.iloc[count]['y_hat']
            anchor_up = df_temp.iloc[count]['y_hat']
            anchor_down = df_temp.iloc[count]['y_hat']
            count += 1
            first_idx += L
            for offset in range(1, L):
                interim_result.iat[mid_idx - offset, -1] = anchor_up + 1
                anchor_up += 1

        # populate RUL for remaining cycles
        for offset in range(1, last_idx - mid_idx + 1):
            interim_result.iat[mid_idx + offset, -1] = anchor_down - 1
            anchor_down -= 1

    return interim_result


################################
#   Global variables
################################

# set upper bound to integrate survival function to find RMST
# kaplan-meier and rsf is highly sensitive to this as they use this to integrate S(t)
rmst_upper_bound = 400

# to re-train untuned baseline NN?
train_untuned_NN = False

# to perform hyperparameter search for NN?
nn_untrended_hyperparameter_tune = False
nn_n_Iterations = 500
train_tuned_NN = False

# to re-train untuned rsf
train_untuned_rsf = False

# to perform hyperparameter search for rsf?
rsf_hyperparameter_tune = False
rsf_n_Iterations = 10  # Number of parameter settings sampled
train_tuned_rsf = False

# to re-train untuned cox-time
train_untuned_ct = False

# If graph should be displayed at the end
show_graph = True

################################
#   Kaplan-Meier Curve
################################

print(delimiter)
print("Started Kaplan-Meier")

# Initilise columns used for training and prediction
train_cols = ['unit num', 'cycle'] + remaining_sensors + ['start', 'breakdown']
predict_cols = ['cycle'] + remaining_sensors + ['start', 'breakdown']  # breakdown value will be 0

km_train = train_org[['unit num', 'cycle', 'breakdown', 'RUL']].groupby('unit num').last()

plt.figure(figsize=(15, 7))
kaplanMeier = KaplanMeierFitter()
kaplanMeier.fit(km_train['cycle'], km_train['breakdown'], label="Survival curve")

# graph plot for report
# fig, ax = plt.subplots()
# kaplanMeier.plot(ax=ax)
# ax.set_xlabel('Cycles', size=15)
# ax.set_ylabel('Probability of survival', size=15)
# ax.legend(loc='upper right', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.show()

# estimate restricted mean survival time from KM curve
km_rmst = restricted_mean_survival_time(kaplanMeier, t=350)
df_result = train_clipped.copy()
print(km_rmst)
df_result['km_rmst'] = km_rmst
km_rmst_arr = [km_rmst for x in range(len(df_result))]
df_result['y_hat'] = km_rmst_arr - df_result['cycle']
df_result['y_hat'].where(df_result['y_hat'] >= 0, 0, inplace=True)
df_master_train['km_rmst'] = df_result['y_hat']
# y_hat = y_hat.clip(upper=clip_level)
result = evaluate("KM_rmst", df_result, 'train')
list_results.append(result)

km_rmst_arr = [km_rmst for x in range(len(test_clipped))]
df_result = test_clipped.copy()
df_result['y_hat'] = km_rmst_arr - df_result['cycle']
df_result['y_hat'].where(df_result['y_hat'] >= 0, 0, inplace=True)  # set negative y_hat to 0
df_result['y_hat'] = df_result['y_hat'].clip(upper=clip_level)
result = evaluate("KM_rmst", df_result, 'test')
list_results.append(result)

df_master_test['km_rmst'] = df_result['y_hat']

################################
#   Cox-PH Model
################################

print(delimiter)
print("Started Cox PH")

# Train Cox model
ctv = CoxTimeVaryingFitter()
ctv.fit(train_org[train_cols], id_col="unit num", event_col='breakdown',
        start_col='start', stop_col='cycle', show_progress=True, step_size=1)

# Calculate log_partial_hazard for all data points
train_cox = train_org.copy()  # need to make a copy so that we can add 'hazard' later
predictions = ctv.predict_log_partial_hazard(train_cox)
train_cox['hazard'] = predictions.to_frame()[0].values

# Fit an exponential curve to the relationship between log_partial_hazard and RUL
popt, pcov = curve_fit(exponential_model, train_cox['hazard'], train_cox['RUL'])

# perform prediction solely on log-partial hazard and evaluate
train_cox['y_hat'] = exponential_model(train_cox['hazard'], *popt)
df_master_train['Cox'] = train_cox['y_hat']
print("fitted exponential curve")
result = evaluate("Cox", train_cox, 'train')
list_results.append(result)

y_pred = ctv.predict_log_partial_hazard(test_clipped)
df_result = test_clipped.copy()
df_result['y_hat'] = exponential_model(y_pred, *popt)
result = evaluate('Cox', df_result, 'test')
list_results.append(result)
df_master_test['Cox'] = df_result['y_hat']

# Plotting for report
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# kaplanMeier.plot(ax=axes[0])
# axes[0].set_xlabel('Cycles', size=15)
# axes[0].set_ylabel('Probability of survival', size=15)
# axes[0].legend(loc='upper right', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# ########################################
# axes[1].scatter(train_cox['hazard'], train_cox['RUL'], label='Individual engine')
# axes[1].plot(range(-15, 20), exponential_model(range(-15, 20), *popt),
#          label='Fitted exponential curve', color='orange')
# axes[1].set_xlabel('Log-partial Hazard', size=15)
# axes[1].set_ylabel('RUL', size=15)
# axes[1].legend(loc='upper right', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.show()

################################
#   Random Forest (Part 1)
################################

# https://towardsdatascience.com/random-forest-for-predictive-maintenance-of-turbofan-engines-5260597e7e8f

print(delimiter)
print("Started Random Forest (Part 1)")

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
df_master_test['RF (pre-tuned)'] = df_result['y_hat']

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
df_master_test['RF (tuned)'] = df_result['y_hat']

#####################################
#   Random Forest (Part 2 - trended)
#####################################

# https://ieeexplore.ieee.org/document/9281004/footnotes#footnotes

print(delimiter)
print("Started Random Forest (Part 2)")

trended_x = train_trend.copy()
trended_y = trended_x.pop('RUL')
trended_x_train, trended_x_val, trended_y_train, trended_y_val = train_test_split(trended_x, trended_y,
                                                                                  test_size=0.25, random_state=6)

new_sensor_features = []
for n in remaining_sensors:
    n = n + "_mean"
    new_sensor_features.append(n)

for n in remaining_sensors:
    n = n + "_trend"
    new_sensor_features.append(n)

# model fitting
rf = RandomForestRegressor(n_estimators=20, criterion="mse", max_features="sqrt", random_state=42)
rf.fit(trended_x_train[new_sensor_features], trended_y_train)

# predict and evaluate, without any hyperparameter tuning
trended_x_val['y_hat'] = rf.predict(trended_x_val[new_sensor_features])
trended_x_val['RUL'] = trended_y_val
print("pre-tuned trended RF")

df_temp = test_trend.copy()
df_temp['y_hat'] = rf.predict(test_trend[new_sensor_features])
df_result = map_test_result(df_temp, test_clipped)
result = evaluate("RF (trended)", df_result, 'test')
list_results.append(result)
df_master_test['RF (trended)'] = df_result['y_hat']

#######################################
#   Neural Network (Part 1 - untrended)
#######################################
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
    nn_untuned = Sequential()
    nn_untuned.add(Dense(16, input_dim=len(remaining_sensors), activation='relu'))
    nn_untuned.add(Dense(32, activation='relu'))
    nn_untuned.add(Dense(64, activation='relu'))
    nn_untuned.add(Dense(1))
    nn_untuned.compile(loss='mean_squared_error', optimizer='adam')

    epochs = 20
    history = nn_untuned.fit(nn_x_train_scaled[remaining_sensors], nn_y_train,
                             validation_data=(nn_x_val_scaled[remaining_sensors], nn_y_val),
                             epochs=epochs, verbose=0)
    nn_untuned.save(filename)  # save trained model

nn_untuned = load_model(filename)
nn_x_val_scaled['y_hat'] = nn_untuned.predict(nn_x_val_scaled[remaining_sensors])
nn_x_val_scaled['RUL'] = nn_y_val
print("pre-tuned Neural Network")
result = evaluate("NN (pre-tuned)", nn_x_val_scaled, 'train')
list_results.append(result)

# nn_x_test_scaled = nn_x_test_scaled.drop(['cycle', 'RUL', 'start'], axis=1)
df_result = test_clipped.copy()
df_result['y_hat'] = nn_untuned.predict(nn_x_test_scaled[remaining_sensors])
result = evaluate("NN (pre-tuned)", df_result, 'test')
list_results.append(result)
df_master_test['NN (pre-tuned)'] = df_result['y_hat']

# Hyperparameter tuning untrended neural network
if nn_untrended_hyperparameter_tune:
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
        nn_untuned = create_model(input_dim, nodes_per_layer, dropout, activation, weights_file)
        # create train-validation split
        gss_search = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)
        for idx_train, idx_val in gss_search.split(nn_x_train, nn_y_train, groups=train_clipped['unit num']):
            X_train_split = nn_x_train.iloc[idx_train].copy()
            y_train_split = nn_y_train.iloc[idx_train].copy()
            X_val_split = nn_x_train.iloc[idx_val].copy()
            y_val_split = nn_y_train.iloc[idx_val].copy()

            # train and evaluate model
            nn_untuned.compile(loss='mean_squared_error', optimizer='adam')
            nn_untuned.load_weights(weights_file)  # reset optimizer and node weights before every training iteration
            history = nn_untuned.fit(X_train_split[remaining_sensors], y_train_split,
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
df_master_test['NN (tuned)'] = df_result['y_hat']

######################################
#   Neural Network (Part 2 - trended)
######################################

# training the model
filename = 'trended_pretuned_NN_model.h5'
train_trended_untuned_NN = True
if train_trended_untuned_NN:
    # construct neural network
    nn_trended_untuned = Sequential()
    nn_trended_untuned.add(Dense(16, input_dim=len(new_sensor_features), activation='relu'))
    nn_trended_untuned.add(Dense(32, activation='relu'))
    nn_trended_untuned.add(Dense(64, activation='relu'))
    nn_trended_untuned.add(Dense(1))
    nn_trended_untuned.compile(loss='mean_squared_error', optimizer='adam')

    epochs = 20
    history = nn_trended_untuned.fit(trended_x_train[new_sensor_features], trended_y_train,
                                     validation_data=(trended_x_val[new_sensor_features], trended_y_val),
                                     epochs=epochs, verbose=0)
    nn_trended_untuned.save(filename)  # save trained model

nn_trended_untuned = load_model(filename)
trended_x_val['y_hat'] = nn_trended_untuned.predict(trended_x_val[new_sensor_features])
trended_x_val['RUL'] = trended_y_val
print("pre-tuned trended Neural Network")
# result = evaluate("NN (pre-tuned trended)", trended_x_val, 'train')
# list_results.append(result)

df_temp = test_trend.copy()
df_temp['y_hat'] = nn_trended_untuned.predict(test_trend[new_sensor_features])
df_result = map_test_result(df_temp, test_clipped)
result = evaluate("NN (pre-tuned trended)", df_result, 'test')
list_results.append(result)
df_master_test['NN (pre-tuned trended)'] = df_result['y_hat']

# Hyperparameter tuning trended neural network
nn_trended_hyperparameter_tune = False
if nn_trended_hyperparameter_tune:
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

    weights_file = 'mlp_trended_hyper_parameter_weights.h5'  # save model weights
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
        nn_x_train, nn_y_train, _ = prep_data(x_train=trended_x,
                                              y_train=trended_y,
                                              x_test=test_trend,
                                              remaining_sensors=new_sensor_features,
                                              lags=specific_lags,
                                              alpha=alpha)
        # create model
        input_dim = len(nn_x_train[new_sensor_features].columns)
        nn_trended_untuned = create_model(input_dim, nodes_per_layer, dropout, activation, weights_file)
        # create train-validation split
        gss_search = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)
        for idx_train, idx_val in gss_search.split(nn_x_train, nn_y_train,
                                                   groups=trended_x['unit num']):
            X_train_split = nn_x_train.iloc[idx_train].copy()
            y_train_split = nn_y_train.iloc[idx_train].copy()
            X_val_split = nn_x_train.iloc[idx_val].copy()
            y_val_split = nn_y_train.iloc[idx_val].copy()

            # train and evaluate model
            nn_trended_untuned.compile(loss='mean_squared_error', optimizer='adam')
            nn_trended_untuned.load_weights(
                weights_file)  # reset optimizer and node weights before every training iteration
            history = nn_trended_untuned.fit(X_train_split[new_sensor_features], y_train_split,
                                             validation_data=(X_val_split[new_sensor_features], y_val_split),
                                             epochs=epochs, batch_size=batch_size, verbose=0)
            mse.append(history.history['val_loss'][-1])

        # append results
        d = {'MSE': np.mean(mse), 'std_MSE': np.std(mse), 'alpha': alpha,
             'epochs': epochs, 'nodes': str(nodes_per_layer), 'dropout': dropout,
             'activation': activation, 'batch_size': batch_size}
        results = results.append(pd.DataFrame(d, index=[0]), ignore_index=True)

    results.to_csv("nn_hyp_trended_results_" + now.replace('/', '-').replace(' ', '_').replace(':', '') + ".csv",
                   index=False)

alpha = 0.2
epochs = 30
specific_lags = [1, 2, 3, 4, 5, 10, 20]
nodes = [64, 128, 256]
dropout = 0.2
activation = 'relu'
batch_size = 16

nn_x_train, nn_y_train, nn_x_test = prep_data(x_train=trended_x,
                                              y_train=trended_y,
                                              x_test=test_trend,
                                              remaining_sensors=new_sensor_features,
                                              lags=specific_lags,
                                              alpha=alpha)
filename = 'finalized_trended_tuned_NN_model.h5'
train_trended_tuned_NN = True
if train_trended_tuned_NN:
    input_dim = len(trended_x_train[new_sensor_features].columns)
    weights_file = 'mlp_trended_hyper_parameter_weights'
    nn_trended_tuned = create_model(input_dim,
                                    nodes_per_layer=nodes,
                                    dropout=dropout,
                                    activation=activation,
                                    weights_file=weights_file)

    nn_trended_tuned.compile(loss='mean_squared_error', optimizer='adam')
    nn_trended_tuned.load_weights(weights_file)
    nn_trended_tuned.fit(trended_x_train[new_sensor_features], trended_y_train,
                         epochs=epochs, batch_size=batch_size, verbose=0)
    nn_trended_tuned.save(filename)  # save trained model

nn_trended_tuned = load_model(filename)
trended_x_val['y_hat'] = nn_trended_tuned.predict(trended_x_val[new_sensor_features])
trended_x_val['RUL'] = trended_y_val
print("tuned trended Neural Network")
# result = evaluate("NN (tuned trended)", trended_x_val, 'train')
# list_results.append(result)

df_temp = test_trend.copy()
df_temp['y_hat'] = nn_trended_tuned.predict(test_trend[new_sensor_features])
df_result = map_test_result(df_temp, test_clipped)
result = evaluate("NN (tuned trended)", df_result, 'test')
list_results.append(result)
df_master_test['NN (tuned trended)'] = df_result['y_hat']

#############################################
#   Neural Network (Part 2 - classification)
#############################################

nb_classes = 16
trended_x = train_trend_class.copy()
trended_x.drop(labels='RUL', axis=1, inplace=True)
trended_y = to_categorical(trended_x.pop('category'), nb_classes)

trended_x_train, trended_x_val, trended_y_train, trended_y_val = train_test_split(trended_x, trended_y,
                                                                                  test_size=0.25, random_state=6)

# training the model
filename = 'trended_classification_NN_model.h5'
train_class_untuned_NN = True
if train_class_untuned_NN:
    # construct neural network
    nn_trended_class_untuned = Sequential()
    nn_trended_class_untuned.add(Dense(16, input_dim=len(new_sensor_features), activation='relu'))
    nn_trended_class_untuned.add(Dense(32, activation='relu'))
    nn_trended_class_untuned.add(Dense(64, activation='relu'))
    nn_trended_class_untuned.add(Dense(nb_classes, activation='softmax'))
    nn_trended_class_untuned.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 20
    history = nn_trended_class_untuned.fit(trended_x_train[new_sensor_features], trended_y_train,
                                           validation_data=(trended_x_val[new_sensor_features], trended_y_val),
                                           epochs=epochs, verbose=0)
    nn_trended_class_untuned.save(filename)  # save trained model

nn_trended_class_untuned = load_model(filename)
# print(np.argmax(nn_trended_class_untuned.predict(trended_x_val[new_sensor_features]), axis=1))
trended_x_val['y_hat'] = np.argmax(nn_trended_class_untuned.predict(trended_x_val[new_sensor_features]),
                                   axis=1)  # get back labels by taking most likely prediction
print("pre-tuned classification Neural Network")

df_temp = test_trend_class.copy()
df_temp['y_hat'] = np.argmax(nn_trended_class_untuned.predict(test_trend_class[new_sensor_features]), axis=1)
df_temp['y_hat'] = df_temp['y_hat'] * 10 + 5
df_result = map_test_result(df_temp, test_clipped)
result = evaluate("NN (trended classification)", df_result, 'test')
list_results.append(result)
df_master_test['NN (trended classification)'] = df_result['y_hat']

# Hyperparameter tuning trended classification neural network
nn_trended_class_hyperparameter_tune = False
if nn_trended_class_hyperparameter_tune:
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

    weights_file = 'mlp_trended_hyper_parameter_weights.h5'  # save model weights
    specific_lags = [1, 2, 3, 4, 5, 10, 20]

    for i in range(ITERATIONS):
        print("Iteration ", str(i + 1))
        cce = []

        # init parameters
        alpha = random.sample(alpha_list, 1)[0]
        epochs = random.sample(epoch_list, 1)[0]
        nodes_per_layer = random.sample(nodes_list, 1)[0]
        dropout = random.sample(dropouts, 1)[0]
        activation = random.sample(activation_functions, 1)[0]
        batch_size = random.sample(batch_size_list, 1)[0]

        # create dataset
        nn_x_train, nn_y_train, _ = prep_data(x_train=trended_x,
                                              y_train=trended_y,
                                              x_test=test_trend,
                                              remaining_sensors=new_sensor_features,
                                              lags=specific_lags,
                                              alpha=alpha)
        nn_y_train = pd.DataFrame(nn_y_train)
        # create model
        input_dim = len(nn_x_train[new_sensor_features].columns)
        nn_trended_class_untuned = create_model(input_dim, nodes_per_layer, dropout,
                                                activation, weights_file, nb_classes, 'classification')
        # create train-validation split
        gss_search = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)
        for idx_train, idx_val in gss_search.split(nn_x_train, nn_y_train,
                                                   groups=trended_x['unit num']):
            X_train_split = nn_x_train.iloc[idx_train].copy()
            y_train_split = nn_y_train.iloc[idx_train].copy()
            X_val_split = nn_x_train.iloc[idx_val].copy()
            y_val_split = nn_y_train.iloc[idx_val].copy()

            # train and evaluate model
            nn_trended_class_untuned.compile(loss='categorical_crossentropy', optimizer='adam')
            nn_trended_class_untuned.load_weights(
                weights_file)  # reset optimizer and node weights before every training iteration
            history = nn_trended_class_untuned.fit(X_train_split[new_sensor_features], y_train_split,
                                                   validation_data=(X_val_split[new_sensor_features], y_val_split),
                                                   epochs=epochs, batch_size=batch_size, verbose=0)
            cce.append(history.history['val_loss'][-1])

        # append results
        d = {'CCE': np.mean(cce), 'alpha': alpha,
             'epochs': epochs, 'nodes': str(nodes_per_layer), 'dropout': dropout,
             'activation': activation, 'batch_size': batch_size}
        results = results.append(pd.DataFrame(d, index=[0]), ignore_index=True)

    results.to_csv("nn_hyp_class_results_" + now.replace('/', '-').replace(' ', '_').replace(':', '') + ".csv",
                   index=False)

alpha = 0.09
epochs = 35
specific_lags = [1, 2, 3, 4, 5, 10, 20]
nodes = [128, 256, 512]
dropout = 0.15
activation = 'relu'
batch_size = 32

nn_x_train, nn_y_train, nn_x_test = prep_data(x_train=trended_x,
                                              y_train=trended_y,
                                              x_test=test_trend,
                                              remaining_sensors=new_sensor_features,
                                              lags=specific_lags,
                                              alpha=alpha)

filename = 'finalized_trended_tuned_class_NN_model.h5'
train_trended_tuned_class_NN = True
if train_trended_tuned_class_NN:
    input_dim = len(trended_x_train[new_sensor_features].columns)
    weights_file = 'mlp_trended_hyper_parameter_weights'
    nn_trended_class_tuned = create_model(input_dim,
                                          nodes_per_layer=nodes,
                                          dropout=dropout,
                                          activation=activation,
                                          weights_file=weights_file,
                                          nb_classes=nb_classes,
                                          type='classification')

    nn_trended_class_tuned.compile(loss='categorical_crossentropy', optimizer='adam')
    nn_trended_class_tuned.load_weights(weights_file)
    nn_trended_class_tuned.fit(trended_x_train[new_sensor_features], trended_y_train,
                               epochs=epochs, batch_size=batch_size, verbose=0)
    nn_trended_class_tuned.save(filename)  # save trained model

nn_trended_class_tuned = load_model(filename)
trended_x_val['y_hat'] = np.argmax(nn_trended_class_tuned.predict(trended_x_val[new_sensor_features]), axis=1)
print("tuned trended class NN")
# result = evaluate("NN (tuned trended)", trended_x_val, 'train')
# list_results.append(result)

df_temp = test_trend.copy()
df_temp['y_hat'] = np.argmax(nn_trended_class_tuned.predict(test_trend[new_sensor_features]), axis=1)
df_temp['y_hat'] = df_temp['y_hat'] * 10 + 5
df_result = map_test_result(df_temp, test_clipped)
result = evaluate("NN (tuned trended classification)", df_result, 'test')
list_results.append(result)
df_master_test['NN (tuned trended classification)'] = df_result['y_hat']

################################
#   Random Survival Forest
################################

print(delimiter)
print("Started Random Survival Forest")

# Data preparation
# cannot use train_clipped as this data is right censored. Hence, last data point of some engines
# is not the breakdown and we need the last RUL of the engine to train the model
rsf_x = train_org.copy()  # hence we are using train_org
# filter for engines that have broken down
rsf_x = rsf_x.reset_index().groupby(by='unit num').last()

# add a balanced mix of engines that have not broken down
# breakdown_0 = train_org.copy()
# breakdown_0 = breakdown_0.loc[breakdown_0['breakdown'] == 0]
# breakdown_0 = breakdown_0.sample(n=100)
# rsf_x = rsf_x.append(breakdown_0, ignore_index=True)  # mix of breakdown and non-breakdown
rsf_x['RUL'] = rsf_x['RUL'].clip(upper=clip_level)
rsf_x['RUL'] = rsf_x.RUL.astype('float')

rsf_y = rsf_x[['breakdown', 'cycle']]
rsf_y['breakdown'].replace(0, False, inplace=True)  # rsf only takes true or false
rsf_y['breakdown'].replace(1, True, inplace=True)  # rsf only takes true or false
rsf_x_train, rsf_x_val, rsf_y_train, rsf_y_val = train_test_split(rsf_x, rsf_y, test_size=0.25,
                                                                  stratify=rsf_y['breakdown'],
                                                                  random_state=7)
# Training RSF
print("Predicting rsf")
rsf_filename = 'finalized_untuned_rsf_model_incl_cycle.sav'
rsf_predict_cols = remaining_sensors.copy()

if train_untuned_rsf:
    from randomsurvivalforest import train_rsf

    train_rsf(rsf_x_train[rsf_predict_cols], rsf_y_train, rsf_filename)
rsf = pickle.load(open(rsf_filename, 'rb'))

result, y_hat = evaluate_rsf("rsf (pre-tuned)", rsf, rsf_x_val, 'train')
list_results.append(result)
rsf_x_val['y_hat'] = y_hat
result, y_hat = evaluate_rsf("rsf (pre-tuned)", rsf, test_clipped, 'test')
list_results.append(result)
df_master_test['rsf (pre-tuned)'] = y_hat

# Hyperparameter tuning RSF
if rsf_hyperparameter_tune:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=1500, num=5)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 1200, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
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

    # perform random search
    rsf_new = RandomSurvivalForest()
    rsf_random = RandomizedSearchCV(estimator=rsf_new,
                                    param_distributions=random_grid,
                                    n_iter=rsf_n_Iterations,  # Number of parameter settings sampled
                                    cv=10,
                                    verbose=10,
                                    random_state=5,
                                    n_jobs=-1)
    # Fit the random search model
    rsf_random.fit(rsf_x[rsf_predict_cols], Surv.from_dataframe('breakdown', 'cycle', rsf_y))
    print(rsf_random.best_params_)

# rsf_grid_search = True
# if rsf_grid_search:
#     # now that we know the best hyperparameter, we perform a proper grid search
#     # Create the parameter grid based on the results of random search
#     param_grid = {
#         'bootstrap': [True],
#         'max_depth': [100, 110, 120, 130],
#         'min_samples_leaf': [3, 4, 5, 6 ,7],
#         'min_samples_split': [3, 4, 5, 6 ,7],
#         'n_estimators': [130, 140, 150, 160, 170]
#     }
#
#     # Create a based model
#     rsf = RandomSurvivalForest()
#
#     # Instantiate the grid search model
#     from sklearn.model_selection import GridSearchCV
#     grid_search = GridSearchCV(estimator=rsf, param_grid=param_grid,
#                                cv=3, n_jobs=-1, verbose=2)
#
#     # Fit the grid search to the data
#     grid_search.fit(rsf_x[remaining_sensors], Surv.from_dataframe('breakdown', 'cycle', rsf_y))
#     print(grid_search.best_params_)

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
df_master_test['rsf (tuned)'] = y_hat

################################
#   Cox-Time Method
################################

# https://jmlr.org/papers/volume20/18-424/18-424.pdf
# https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html

# print(delimiter)
# print("Started CoxTime Method")
#
# np.random.seed(1234)
# _ = torch.manual_seed(123)
#
# # Prepare data
# ct_x = rsf_x.copy()  # already grouped by unit num
# ct_y = rsf_y.copy()
#
# ct_x_scaled, ct_test_scaled = minmax_scaler(ct_x, test_clipped, remaining_sensors)
# gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
# ct_x_train, ct_x_val, ct_y_train, ct_y_val = train_test_split(ct_x_scaled, ct_y,
#                                                               test_size=0.25, random_state=7)
#
# labtrans = CoxTime.label_transform()
# get_target = lambda df: (df['cycle'].values.astype('float32'), df['breakdown'].values.astype('float32'))
# ct_y_train_split_tuple = labtrans.fit_transform(*get_target(ct_y_train))
# ct_y_val_split_tuple = labtrans.transform(*get_target(ct_y_val))
# val = tt.tuplefy(ct_x_val[remaining_sensors].to_numpy().astype('float32'), ct_y_val_split_tuple)
#
# # Prepare model
# ct_untuned_filename = 'finalized_untuned_ct_model.h5'
# if train_untuned_ct:
#     in_features = ct_x_train[remaining_sensors].shape[1]
#     num_nodes = [32, 32]
#     batch_norm = True
#     dropout = 0.1
#     batch_size = 256
#
#     net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
#     nn_untuned = CoxTime(net, tt.optim.Adam,
#                          labtrans=labtrans)  # set labtrans to get back the correct time scale in output
#
#     lrfinder = nn_untuned.lr_finder(ct_x_train[remaining_sensors].to_numpy().astype('float32'),
#                                     ct_y_train_split_tuple, batch_size, tolerance=2)
#     # _ = lrfinder.plot()
#     # plt.show()
#     #
#     # print("The best learning rate is: ", str(lrfinder.get_best_lr()))
#
#     nn_untuned.optimizer.set_lr(lrfinder.get_best_lr())
#
#     log = nn_untuned.fit(ct_x_train[remaining_sensors].to_numpy().astype('float32'), ct_y_train_split_tuple,
#                          batch_size,
#                          epochs=512,
#                          callbacks=[tt.callbacks.EarlyStopping()],
#                          verbose=True,
#                          val_data=val.repeat(10).cat())
# #     pickle.dump(log, open(ct_untuned_filename, 'wb'))
# # ct_untuned = pickle.load(open(ct_untuned_filename, 'rb'))
# # _ = log.plot()
# # plt.show()
#
# # print(nn_untuned.partial_log_likelihood(*val).mean())
# _ = nn_untuned.compute_baseline_hazards()
# surv = nn_untuned.predict_surv_df(ct_x_val[remaining_sensors].to_numpy().astype('float32'))
#
# # print(test_clipped.shape)
# # print(surv)
# # surv.iloc[:, :20].plot()
# # plt.ylabel('S(t | x)')
# # _ = plt.xlabel('Time')
# # plt.show()
#
# ct_rmst = []  # create a list to store the rmst of each survival curve
# for col in surv:
#     km_rmst = restricted_mean_survival_time(surv[col].to_frame(), t=400)  # calculate restricted mean survival time
#     ct_rmst.append(km_rmst)
#
# ct_x_val['y_hat'] = ct_rmst - ct_x_val['cycle']
# # ct_x_val['y_hat'].where(ct_x_val['y_hat'] >= 0, 0, inplace=True)
# result = evaluate("ct (untuned)", ct_x_val, 'train')
# list_results.append(result)
#
# surv = nn_untuned.predict_surv_df(ct_test_scaled[remaining_sensors].to_numpy().astype('float32'))
#
# ct_rmst = []  # create a list to store the rmst of each survival curve
# for col in surv:
#     km_rmst = restricted_mean_survival_time(surv[col].to_frame(), t=400)  # calculate restricted mean survival time
#     ct_rmst.append(km_rmst)
#
# ct_test_scaled['y_hat'] = ct_rmst - ct_test_scaled['cycle']
# # ct_test_scaled['y_hat'].where(ct_x_val['y_hat'] >= 0, 0, inplace=True)
# result = evaluate("ct (untuned)", ct_test_scaled, 'test')
# ct_test_scaled['y_hat'] = ct_test_scaled['y_hat'].clip(upper=clip_level)
# list_results.append(result)
# graph_data["ct (untuned)"] = ct_test_scaled['y_hat']

# Hyperparameter tuning Cox-Time
# ct_hyperparameter_tune = True
# if ct_hyperparameter_tune:
#     # Number of trees in random forest
#     n_estimators = [int(x) for x in np.linspace(start=50, stop=150, num=5)]
#     # Maximum number of levels in tree
#     max_depth = [int(x) for x in np.linspace(10, 120, num=10)]
#     # Number of features to consider at every split
#     max_features = ['auto', 'sqrt']
#     # Minimum number of samples required to split a node
#     min_samples_split = [int(x) for x in np.linspace(start=5, stop=30, num=1)]
#     # Minimum number of samples required at each leaf node
#     min_samples_leaf = [int(x) for x in np.linspace(start=5, stop=30, num=1)]
#
#     # Create the random grid
#     random_grid = {'n_estimators': n_estimators,
#                    'max_features': max_features,
#                    'max_depth': max_depth,
#                    'min_samples_split': min_samples_split,
#                    'min_samples_leaf': min_samples_leaf}
#
#     # perform random search
#     rsf = RandomSurvivalForest()

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
filename = full_path + "saved_results_" + now.replace('/', '-').replace(' ', '_').replace(':', '') + ".xlsx"
df_results.to_excel(filename, index=False)
df_master_test.to_csv("master_test.csv", index=False)
df_master_train.to_csv("master_train.csv", index=False)
