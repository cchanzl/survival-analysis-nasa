import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress info, warning and error tensorflow messages

from datetime import datetime  # to timestamp results of each model

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

import pandas as pd
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

##########################
#   Loading Data
##########################

delimiter = "*" * 40
header = ["unit num", "cycle", "op1", "op2", "op3"]
for i in range(0, 26):
    name = "sens"
    name = name + str(i + 1)
    header.append(name)

full_path = "C:/Users/chanzl_thinkpad/Dropbox/Imperial/Individual Project/NASA/Dataset/"
y_test = pd.read_csv(full_path + 'RUL_FD001.txt', delimiter=" ", header=None)
df_train_001 = pd.read_csv(full_path + 'train_FD001.txt', delimiter=" ", names=header)
x_test_org = pd.read_csv(full_path + 'test_FD001.txt', delimiter=" ", names=header)

# create df to append result of each model
evaluation_metrics = ["RMSE", "CI_SK", "R2"]
results_header = ["model_name", "train_test"] + evaluation_metrics
list_results = []


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


def evaluate(model, y_true, y_hat, breakdown, label='test'):
    breakdown.replace(0, False, inplace=True)  # rsf only takes true or false
    breakdown.replace(1, True, inplace=True)  # rsf only takes true or false

    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)

    # the concordance index is interested on the order of the predictions, not the predictions themselves
    # https://medium.com/analytics-vidhya/concordance-index-72298c11eac7#:~:text=The%20concordance%20index%20or%20c,this%20definition%20mean%20in%20practice
    ci_sk = ci_scikit(breakdown.to_numpy(), y_true, y_hat)[0]

    print('{} set RMSE:{:.2f}, CI(scikit):{:.4f}, R2:{:.2f}'.format(label, rmse, ci_sk, variance))
    result = [model, label, rmse, ci_sk, variance]
    return result


def exponential_model(z, a, b):
    return a * np.exp(-b * z)


def train_val_group_split(X, y, gss, groups, print_groups=False):
    for idx_train, idx_val in gss.split(X, y, groups=groups):
        if print_groups:
            print('train_split_engines', train_clipped_dropped.iloc[idx_train]['unit num'].unique(), '\n')
            print('validate_split_engines', train_clipped_dropped.iloc[idx_val]['unit num'].unique(), '\n')

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
                                                 columns=sensor_names)

    # scale test set
    df_test_scaled = df_test.copy()
    # df_test_scaled = df_test_scaled.drop('cycle', axis=1).groupby('unit num').last().copy()
    df_test_scaled[sensor_names] = pd.DataFrame(scaler.transform(df_test_scaled[sensor_names]),
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

    X_train_interim = add_specific_lags(X_train_interim, lags, remaining_sensors)
    X_test_interim = add_specific_lags(X_test_interim, lags, remaining_sensors)

    X_train_breakdown = X_train_interim['breakdown']
    X_train_interim.drop(['unit num', 'cycle', 'RUL', 'breakdown', 'start'], axis=1, inplace=True)
    X_test_interim.drop(['unit num', 'cycle', 'RUL', 'breakdown', 'start'], axis=1, inplace=True)

    train_idx = X_train_interim.index
    y_train = y_train.iloc[train_idx]

    return X_train_interim, y_train, X_test_interim, train_idx, X_train_breakdown


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

# apply a floor to RUL. 125 is selected as the min cycle is 128. See EDA.
train_clipped = train_org.copy()
train_clipped['RUL'] = train_clipped['RUL'].clip(upper=125)
test_clipped = test_org.copy()
test_clipped['RUL'] = test_clipped['RUL'].clip(upper=125)

# Apply cut-off
cut_off = 200  # Important to improve accuracy
train_clipped = train_clipped[train_clipped['cycle'] <= cut_off].copy()
test_clipped = test_clipped[test_clipped['cycle'] <= cut_off].copy()
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(train_clipped[['unit num', 'cycle', 'RUL']])

# Initilise columns used for training and prediction
train_cols = ['unit num', 'cycle'] + remaining_sensors + ['start', 'breakdown']
predict_cols = ['cycle'] + remaining_sensors + ['start', 'breakdown']  # breakdown value will be 0

################################
#   Kaplan-Meier Curve
################################

print(delimiter)
print("Started Kaplan-Meier")

km_train = train_clipped[['unit num', 'cycle', 'breakdown', 'RUL']].groupby('unit num').last()

plt.figure(figsize=(15, 7))
kaplanMeier = KaplanMeierFitter()
kaplanMeier.fit(km_train['cycle'], km_train['breakdown'])

# kaplanMeier.plot()
# plt.ylabel("Probability of survival")
# plt.show()
# plt.close()

# estimate restricted mean survival time from KM curve
km_rmst = restricted_mean_survival_time(kaplanMeier, t=cut_off)
km_train['km_rmst'] = km_rmst

result = evaluate("KM_rmst", km_train['RUL'], km_train['km_rmst'], km_train['breakdown'], 'train')
list_results.append(result)

km_rmst_arr = [km_rmst for x in range(len(test_clipped))]
y_hat = km_rmst_arr - test_clipped['cycle']
result = evaluate("KM_rmst", test_clipped['RUL'], y_hat, test_clipped['breakdown'], 'test')
list_results.append(result)

################################
#   Cox-PH Model
################################

print(delimiter)
print("Started Cox PH")

# Train Cox model
ctv = CoxTimeVaryingFitter()
ctv.fit(train_clipped[train_cols], id_col="unit num", event_col='breakdown',
        start_col='start', stop_col='cycle', show_progress=True, step_size=1)
# ctv.print_summary()
# plt.figure(figsize=(10, 5))
# ctv.plot()
# plt.show()

# get engines from dataset which are still functioning but right censored to predict their RUL
df = train_clipped.groupby("unit num").last()  # get the last entry of each engine unit
train_log_ph = df[df['breakdown'] == 0].copy()
train_log_ph.reset_index(inplace=True)

# Predict log_partial_hazard for engines that have not broke down
predictions = ctv.predict_log_partial_hazard(train_log_ph[predict_cols])
predictions = predictions.to_frame()
predictions.rename(columns={0: "predictions"}, inplace=True)

predictions['unit num'] = train_log_ph['unit num']
predictions['RUL'] = train_log_ph['RUL']

# plt.figure(figsize=(15, 5))
# plt.plot(predictions['RUL'], predictions['predictions'], '.b')
# xlim = plt.gca().get_xlim()
# plt.xlim(xlim[1], xlim[0])
# plt.xlabel('RUL')
# plt.ylabel('log_partial_hazard')
# plt.show()

# Plot log_partial_hazard against RUL to see trend
train_log_ph.set_index('unit num', inplace=True)
X = train_clipped.loc[train_clipped['unit num'].isin(train_log_ph.index)]
X_unique = len(X['unit num'].unique())

# plt.figure(figsize=(15, 5))
# for i in range(1, X_unique, 2):
#     X_sub = X.loc[X['unit num'] == i]
#     if X_sub.empty:
#         continue
#     predictions = ctv.predict_partial_hazard(X_sub)
#     predictions = predictions.to_frame()[0].values
#     plt.plot(X_sub['cycle'].values, np.log(predictions))
# plt.xlabel('cycle')
# plt.ylabel('log_partial_hazard')
# plt.show()

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
y_hat = exponential_model(train_cox['hazard'], *popt)
print("fitted exponential curve")
result = evaluate("Cox", train_cox['RUL'], y_hat, train_cox['breakdown'], 'train')
list_results.append(result)

y_pred = ctv.predict_log_partial_hazard(test_clipped)
y_hat = exponential_model(y_pred, *popt)
result = evaluate('Cox', test_clipped['RUL'], y_hat, test_clipped['breakdown'], 'test')
list_results.append(result)

################################
#   Random Forest
################################

# https://towardsdatascience.com/random-forest-for-predictive-maintenance-of-turbofan-engines-5260597e7e8f

print(delimiter)
print("Started Random Forest")

rf = RandomForestRegressor(n_estimators=100, criterion="mse", max_features="sqrt", random_state=42)
rf_x = train_clipped.copy()
rf_y = rf_x.pop('RUL')
rf_x_train, rf_x_val, rf_y_train, rf_y_val = train_test_split(rf_x, rf_y, test_size=0.25)
rf_x_train_dropped = rf_x_train.drop(['cycle', 'unit num', 'breakdown', 'start'], axis=1)
rf_x_val_dropped = rf_x_val.drop(['cycle', 'unit num', 'breakdown', 'start'], axis=1)
rf.fit(rf_x_train_dropped, rf_y_train)

# predict and evaluate, without any hyperparameter tuning
y_hat_val = rf.predict(rf_x_val_dropped)
print("pre-tuned RF")
result = evaluate('RF (pre-tuned)', rf_y_val, y_hat_val, rf_x_val['breakdown'], 'train')
list_results.append(result)

x_test_clipped_dropped = test_clipped.drop(['cycle', 'unit num', 'breakdown', 'start', 'RUL'], axis=1)
y_hat_test = rf.predict(x_test_clipped_dropped)
result = evaluate("RF (pre-tuned)", test_clipped['RUL'], y_hat_test, test_clipped['breakdown'], 'test')
list_results.append(result)

# perform some checks on layout of a SINGLE tree
# print(rf.estimators_[5].tree_.max_depth)  # check how many nodes in the longest path
# rf.estimators_[5].tree_.n_node_samples    # check how many samples in the last nodes

# crudely tweaked random forest
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42,
                           max_depth=8, min_samples_leaf=50)
rf.fit(rf_x_train_dropped, rf_y_train)

# predict and evaluate
y_hat_val = rf.predict(rf_x_val_dropped)
print("crudely tuned RF")
result = evaluate('RF (tuned)', rf_y_val, y_hat_val, rf_x_val['breakdown'], 'train')
list_results.append(result)

y_hat_test = rf.predict(x_test_clipped_dropped)
result = evaluate("RF (tuned)", test_clipped['RUL'], y_hat_test, test_clipped['breakdown'], 'test')
# to_compare = y_test.copy()
# to_compare["pred"] = y_hat_test.tolist()
# print(to_compare)
list_results.append(result)

################################
#   Neural Network
################################

# https://towardsdatascience.com/lagged-mlp-for-predictive-maintenance-of-turbofan-engines-c79f02a15329

print(delimiter)
print("Started Neural Network")
# make a copy of the original full dataset as we need to scale it for NN
train_NN = train_org.copy()  # why cannot use clipped data?

# scaling using minmax
nn_y_train = train_NN.pop('RUL')
nn_x_train_scaled, nn_x_test_scaled = minmax_scaler(train_NN, test_org, remaining_sensors)

# split scaled dataset into train test split
gss = GroupShuffleSplit(n_splits=1, train_size=0.80,
                        random_state=42)  # even though we set np and tf seeds, gss requires its own seed
nn_x_train_scaled, nn_y_train, nn_x_val_scaled, nn_y_val = train_val_group_split(nn_x_train_scaled,
                                                                                 nn_y_train, gss,
                                                                                 nn_x_train_scaled['unit num'])

# training the model
train = False
filename = 'finalized_pretuned_NN_model.h5'
if train:
    # construct neural network
    model = Sequential()
    model.add(Dense(16, input_dim=len(remaining_sensors), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    epochs = 20
    history = model.fit(nn_x_train_scaled[remaining_sensors], nn_y_train,
                        validation_data=(nn_x_val_scaled[remaining_sensors], nn_y_val),
                        epochs=epochs, verbose=0)
    model.save(filename)  # save trained model

model = load_model(filename)
y_hat_val = model.predict(nn_x_val_scaled[remaining_sensors])
print("pre-tuned Neural Network")
result = evaluate("NN (pre-tuned)", nn_y_val, y_hat_val.flatten(), nn_x_val_scaled['breakdown'], 'train')
list_results.append(result)

# nn_x_test_scaled = nn_x_test_scaled.drop(['cycle', 'RUL', 'start'], axis=1)
y_hat_test = model.predict(nn_x_test_scaled[remaining_sensors])
result = evaluate("NN (pre-tuned)", test_org['RUL'], y_hat_test.flatten(), test_org['breakdown'], 'test')
list_results.append(result)

# hyperparameter tuning
tune = False
if tune:
    alpha_list = list(np.arange(5, 20 + 1, 0.5) / 100)
    epoch_list = list(np.arange(10, 50 + 1, 5))
    nodes_list = [[8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512]]

    # lowest dropout=0.1, because I know zero dropout will yield better training results but worse generalization (overfitting)
    dropouts = list(np.arange(0, 4 + 1, 0.5) / 10)

    # earlier testing revealed relu performed significantly worse, so I removed it from the options
    activation_functions = ['tanh', 'sigmoid']
    batch_size_list = [16, 32, 64, 128, 256, 512]

    ITERATIONS = 10
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
        nn_x_train, nn_y_train, _, idx, _ = prep_data(x_train=train_org,
                                                      y_train=train_org['RUL'],
                                                      x_test=test_org,
                                                      remaining_sensors=remaining_sensors,
                                                      lags=specific_lags,
                                                      alpha=alpha)
        # create model
        input_dim = len(nn_x_train.columns)
        model = create_model(input_dim, nodes_per_layer, dropout, activation, weights_file)

        # create train-validation split
        gss_search = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)
        for idx_train, idx_val in gss_search.split(nn_x_train, nn_y_train, groups=train_org.iloc[idx]['unit num']):
            X_train_split = nn_x_train.iloc[idx_train].copy()
            y_train_split = nn_y_train.iloc[idx_train].copy()
            X_val_split = nn_x_train.iloc[idx_val].copy()
            y_val_split = nn_y_train.iloc[idx_val].copy()

            # train and evaluate model
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.load_weights(weights_file)  # reset optimizer and node weights before every training iteration
            history = model.fit(X_train_split, y_train_split,
                                validation_data=(X_val_split, y_val_split),
                                epochs=epochs, batch_size=batch_size, verbose=0)
            mse.append(history.history['val_loss'][-1])

        # append results
        d = {'MSE': np.mean(mse), 'std_MSE': np.std(mse), 'alpha': alpha,
             'epochs': epochs, 'nodes': str(nodes_per_layer), 'dropout': dropout,
             'activation': activation, 'batch_size': batch_size}
        results = results.append(pd.DataFrame(d, index=[0]), ignore_index=True)

    results.to_csv("nn_hyp_results_" + now.replace('/', '-').replace(' ', '_').replace(':', '') + ".csv", index=False)

alpha = 0.05
epochs = 30
specific_lags = [1, 2, 3, 4, 5, 10, 20]
nodes = [64, 128, 256]
dropout = 0.2
activation = 'tanh'
batch_size = 64

nn_x_train, nn_y_train, nn_x_test, _, df_breakdown = prep_data(x_train=train_org,
                                                               y_train=train_org['RUL'],
                                                               x_test=test_org,
                                                               remaining_sensors=remaining_sensors,
                                                               lags=specific_lags,
                                                               alpha=alpha)
train = False
filename = 'finalized_tuned_NN_model.h5'
if train:
    input_dim = len(nn_x_train.columns)
    weights_file = 'mlp_hyper_parameter_weights'
    nn_lagged_tuned = create_model(input_dim,
                                   nodes_per_layer=nodes,
                                   dropout=dropout,
                                   activation=activation,
                                   weights_file=weights_file)

    nn_lagged_tuned.compile(loss='mean_squared_error', optimizer='adam')
    nn_lagged_tuned.load_weights(weights_file)
    nn_lagged_tuned.fit(nn_x_train, nn_y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    nn_lagged_tuned.save(filename)  # save trained model

# predict and evaluate
nn_lagged_tuned = load_model(filename)
print("tuned Neural Network")
y_hat_val = nn_lagged_tuned.predict(nn_x_train)
result = evaluate("NN (lagged+tuned)", nn_y_train, y_hat_val.flatten(), df_breakdown, 'train')
list_results.append(result)

test_idx = nn_x_test.index
y_hat_test = nn_lagged_tuned.predict(nn_x_test)
result = evaluate("NN (lagged+tuned)", test_org.iloc[test_idx]['RUL'], y_hat_test.flatten(),
                  test_org.iloc[test_idx]['breakdown'], 'test')
list_results.append(result)

################################
#   Random Survival Forest
################################

print(delimiter)
print("Started Random Survival Forest")

# Data preparation
rsf_x = train_clipped.copy()
rsf_x['RUL'] = rsf_x.RUL.astype('float')

rsf_y = rsf_x[['breakdown', 'cycle', 'RUL']]
rsf_y['breakdown'].replace(0, False, inplace=True)  # rsf only takes true or false
rsf_y['breakdown'].replace(1, True, inplace=True)  # rsf only takes true or false

rsf_x_train, rsf_x_val, rsf_y_train, rsf_y_val = train_test_split(rsf_x, rsf_y, test_size=0.25)

# Predicting RSF
print("Predicting rsf")
rsf_filename = 'finalized_rsf_model.sav'
rsf = pickle.load(open(rsf_filename, 'rb'))


# Estimate remaining useful life
def evaluate_rsf(name, rsf, rsf_x, rsf_y, label):
    # attribute_dropped = ['cycle', 'breakdown', 'start']  # there is no 'RUL' and 'unit num' for x_test
    # if label != 'test':
    attribute_dropped = ['unit num', 'cycle', 'RUL', 'breakdown', 'start']
    rsf_x_dropped = rsf_x.drop(attribute_dropped, axis=1)
    rsf_y_dropped = rsf_y.drop('RUL', axis=1)
    surv = rsf.predict_survival_function(rsf_x_dropped, return_array=True)  # return S(t) for each data point

    rsf_rmst = []  # create a list to store the rmst of each survival curve

    for i, s in enumerate(surv):
        # plt.step(rsf.event_times_, s, where="post", label=str(i))

        # calculate rmst
        df_s = pd.DataFrame(s)
        df_s.set_index(rsf.event_times_, inplace=True)
        km_rmst = restricted_mean_survival_time(df_s, t=cut_off)  # calculate restricted mean survival time
        rsf_rmst.append(km_rmst)

    # plt.ylabel("Survival probability")
    # plt.xlabel("Time in cycles")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    rsf_RUL = rsf_rmst - rsf_x['cycle']
    # print('The C-index (scikit-survival) is ', ci_scikit(rsf_y['breakdown'], rsf_y['RUL'], rsf_RUL)[0])
    # print('The C-index (worse being nearer to 1) is ',
    #      rsf.score(rsf_x_dropped, Surv.from_dataframe('breakdown', 'cycle', rsf_y_dropped)))

    result_interim = evaluate(name, rsf_y['RUL'], rsf_RUL, rsf_y['breakdown'], label)
    return result_interim


result = evaluate_rsf("rsf (pre-tuned)", rsf, rsf_x_val, rsf_y_val, 'train')
list_results.append(result)

rsf_test_y = test_clipped[['breakdown', 'cycle', 'RUL']]
rsf_test_y['breakdown'].replace(0, False, inplace=True)  # rsf only takes true or false
rsf_test_y['breakdown'].replace(1, True, inplace=True)  # rsf only takes true or false
result = evaluate_rsf("rsf (pre-tuned)", rsf, test_clipped, rsf_test_y, 'test')
list_results.append(result)

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
    full_path = "C:/Users/chanzl_thinkpad/Dropbox/Imperial/Individual Project/NASA/survival-analysis-nasa/results/"
    filename = full_path + "saved_results_" + now.replace('/', '-').replace(' ', '_').replace(':', '') + ".csv"
    df_results.to_csv(filename, index=False)
