import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress info, warning and error tensorflow messages

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

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
results_header = ["model_name", "train_test", "RMSE", "R2"]
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


def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{:.2f}, R2:{:.2f}'.format(label, rmse, variance))
    return label, rmse, variance


def exponential_model(z, a, b):
    return a * np.exp(-b * z)


def train_val_group_split(X, y, gss, groups, print_groups=True):
    for idx_train, idx_val in gss.split(X, y, groups=groups):
        # if print_groups:
        # print('train_split_engines', train_clipped_dropped.iloc[idx_train]['unit num'].unique(), '\n')
        # print('validate_split_engines', train_clipped_dropped.iloc[idx_val]['unit num'].unique(), '\n')

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


def prep_data(df_train, train_label, df_test, remaining_sensors, lags, alpha, n=0):
    X_train_interim, X_test_interim = minmax_scaler(df_train, df_test, remaining_sensors)

    X_train_interim = exponential_smoothing(X_train_interim, remaining_sensors, n, alpha)
    X_test_interim = exponential_smoothing(X_test_interim, remaining_sensors, n, alpha)

    X_train_interim = add_specific_lags(X_train_interim, lags, remaining_sensors)
    X_test_interim = add_specific_lags(X_test_interim, lags, remaining_sensors)

    X_train_interim.drop(['op1', 'op2', 'op3', 'unit num', 'cycle', 'RUL'], axis=1, inplace=True)
    X_test_interim.drop(['op1', 'op2', 'op3', 'cycle'], axis=1, inplace=True)
    X_test_interim = X_test_interim.groupby('unit num').last().copy()

    idx = X_train_interim.index
    train_label = train_label.iloc[idx]

    return X_train_interim, train_label, X_test_interim, idx


################################
#   General Data pre-processing
################################

print(delimiter)
print("Started data preprocessing")

# add RUL column
train_org = add_remaining_useful_life(df_train_001)

# apply a floor to RUL. 125 is selected as the min cycle is 128. See EDA.
train_clipped = train_org.copy()
train_clipped['RUL'].clip(upper=125, inplace=True)

# Based on MannKendall, sensor 2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20 and 21 are selected
drop_sensors = ['sens1', 'sens5', 'sens6', 'sens9', 'sens10', 'sens14',
                'sens16', 'sens18', 'sens19', 'sens22', 'sens23', 'sens24', 'sens25', 'sens26']

drop_labels = drop_sensors + ["op1", "op2", "op3"]

remaining_sensors = ['sens2', 'sens3', 'sens4', 'sens7', 'sens8', 'sens11',
                     'sens12', 'sens13', 'sens15', 'sens17', 'sens20', 'sens21']

all_sensors = drop_sensors + remaining_sensors

train_clipped_dropped = train_clipped.copy()
train_clipped_dropped.drop(labels=drop_labels, axis=1, inplace=True)

# Add event indicator column
train_clipped_dropped['breakdown'] = 0
idx_last_record = train_clipped_dropped.reset_index().groupby(by='unit num')[
    'index'].last()  # engines breakdown at the last cycle
train_clipped_dropped.at[idx_last_record, 'breakdown'] = 1

# Add start cycle column
train_clipped_dropped['start'] = train_clipped_dropped['cycle'] - 1

# Apply cut-off
cut_off = 200  # Important to improve accuracy
train_censored = train_clipped_dropped[train_clipped_dropped['cycle'] <= cut_off].copy()  # Final dataset to use for all model

# prep test set
x_test_dropped = x_test_org.copy()
x_test_dropped.drop(labels=drop_labels, axis=1, inplace=True)
x_test_dropped['breakdown'] = 0
x_test_dropped['start'] = x_test_dropped['cycle'] - 1
y_test.drop(y_test.columns[1], axis=1, inplace=True)

# Initilise columns used for training and prediction
train_cols = ['unit num', 'cycle'] + remaining_sensors + ['start', 'breakdown']
predict_cols = ['cycle'] + remaining_sensors + ['start', 'breakdown']  # breakdown value will be 0

################################
#   Kaplan-Meier Curve
################################

print(delimiter)
print("Started Kaplan-Meier")

data = train_censored[['unit num', 'cycle', 'breakdown', 'RUL']].groupby('unit num').last()

plt.figure(figsize=(15, 7))
kaplanMeier = KaplanMeierFitter()
kaplanMeier.fit(data['cycle'], data['breakdown'])

# kaplanMeier.plot()
# plt.ylabel("Probability of survival")
# plt.show()
# plt.close()

# estimate restricted mean survival time from KM curve
km_rmst = restricted_mean_survival_time(kaplanMeier, t=cut_off)
data['km_rmst'] = km_rmst

label, rmse, variance = evaluate(data['RUL'], data['km_rmst'], 'train')
KM_train = ["KM_rmst", label, rmse, variance]
list_results.append(KM_train)

y_hat = [km_rmst for x in range(len(y_test))]
label, rmse, variance = evaluate(y_test, y_hat)
KM_test = ["KM_rmst", label, rmse, variance]
list_results.append(KM_test)

################################
#   Cox-PH Model
################################

print(delimiter)
print("Started Cox PH")

# Train Cox model
ctv = CoxTimeVaryingFitter()
ctv.fit(train_censored[train_cols], id_col="unit num", event_col='breakdown',
        start_col='start', stop_col='cycle', show_progress=True, step_size=1)
# ctv.print_summary()
# plt.figure(figsize=(10, 5))
# ctv.plot()
# plt.show()

# get engines from dataset which are still functioning but right censored to predict their RUL
df = train_censored.groupby("unit num").last()  # get the last entry of each engine unit
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
X = train_clipped_dropped.loc[train_clipped_dropped['unit num'].isin(train_log_ph.index)]
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
train_cox = train_censored.copy()
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
label, rmse, variance = evaluate(train_cox['RUL'], y_hat, 'train')
Cox_train = ["Cox", label, rmse, variance]
list_results.append(Cox_train)

y_pred = ctv.predict_log_partial_hazard(x_test_dropped.groupby('unit num').last().reset_index())
y_hat = exponential_model(y_pred, *popt)
label, rmse, variance = evaluate(y_test, y_hat)
Cox_test = ["Cox", label, rmse, variance]
list_results.append(Cox_test)

################################
#   Random Forest
################################

# https://towardsdatascience.com/random-forest-for-predictive-maintenance-of-turbofan-engines-5260597e7e8f

print(delimiter)
print("Started Random Forest")

rf = RandomForestRegressor(n_estimators=100, criterion="mse", max_features="sqrt", random_state=42)
x_train_rf = train_censored.copy()
x_train_rf = x_train_rf.drop('cycle', axis=1)
y_train_rf = x_train_rf.pop('RUL')
rf.fit(x_train_rf, y_train_rf)

# predict and evaluate, without any hyperparameter tuning
y_hat_train = rf.predict(x_train_rf)
print("pre-tuned RF")
label, rmse, variance = evaluate(y_train_rf, y_hat_train, 'train')
RF_train = ["RF (pre-tuned)", label, rmse, variance]
list_results.append(RF_train)

y_hat_test = rf.predict(x_test_dropped.groupby('unit num').last())
label, rmse, variance = evaluate(y_test, y_hat_test)
RF_test = ["RF (pre-tuned)", label, rmse, variance]
list_results.append(RF_test)

# perform some checks on layout of a SINGLE tree
# print(rf.estimators_[5].tree_.max_depth)  # check how many nodes in the longest path
# rf.estimators_[5].tree_.n_node_samples    # check how many samples in the last nodes

# crudely tweaked random forest
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42,
                           max_depth=8, min_samples_leaf=50)
rf.fit(x_train_rf, y_train_rf)

# predict and evaluate
y_hat_train = rf.predict(x_train_rf)
print("crudely tuned RF")
label, rmse, variance = evaluate(y_train_rf, y_hat_train, 'train')
RF_train = ["RF (tuned)", label, rmse, variance]
list_results.append(RF_train)

y_hat_test = rf.predict(x_test_dropped.groupby('unit num').last())
label, rmse, variance = evaluate(y_test, y_hat_test)
RF_test = ["RF (tuned)", label, rmse, variance]
list_results.append(RF_test)

################################
#   Neural Network
################################

# https://towardsdatascience.com/lagged-mlp-for-predictive-maintenance-of-turbofan-engines-c79f02a15329

print(delimiter)
print("Started Neural Network")
print("check the data set entering the NN is using censored and has the correct columns")
# preprocess data by performing scaling using MinMax on the original dataset before dropping, clipping and censoring
train_NN = train_org.copy()
attribute_dropped = ['cycle', 'op1', 'op2', 'op3']
train_NN = train_NN.drop(attribute_dropped, axis=1)
x_train_NN_scaled, x_test_NN_scaled = minmax_scaler(train_NN, x_test_org, all_sensors)
y_train_NN_scaled = x_train_NN_scaled.pop('RUL')

# split scaled dataset into train test split
gss = GroupShuffleSplit(n_splits=1, train_size=0.80,
                        random_state=42)  # even though we set np and tf seeds, gss requires its own seed
split_result = train_val_group_split(x_train_NN_scaled, y_train_NN_scaled, gss, train_clipped_dropped['unit num'],
                                     print_groups=True)
X_train_split_scaled, y_train_split_scaled, X_val_split_scaled, y_val_split_scaled = split_result

train_cols_NN = remaining_sensors  # select columns used for trg in NN
input_dim = len(train_cols_NN)

# training the model
train = False
filename = 'finalized_pretuned_NN_model.h5'
if train:
    # construct neural network
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    epochs = 20
    history = model.fit(x_train_NN_scaled[train_cols_NN], y_train_NN_scaled,
                        validation_data=(X_val_split_scaled[train_cols_NN], y_val_split_scaled),
                        epochs=epochs, verbose=0)
    model.save(filename)  # save trained model

model = load_model(filename)
y_hat_train = model.predict(x_train_NN_scaled[train_cols_NN])
print("pre-tuned Neural Network")
label, rmse, variance = evaluate(y_train_NN_scaled, y_hat_train, 'train')
NN_train = ["NN (pre-tuned)", label, rmse, variance]
list_results.append(NN_train)

x_test_NN_scaled = x_test_NN_scaled.drop('cycle', axis=1).groupby('unit num').last().copy()
y_hat_test = model.predict(x_test_NN_scaled[train_cols_NN])
label, rmse, variance = evaluate(y_test, y_hat_test)
NN_test = ["NN (pre-tuned)", label, rmse, variance]
list_results.append(NN_test)

# hyperparameter tuning
tune = False
if tune:
    alpha_list = [0.01, 0.05] + list(np.arange(10, 60 + 1, 10) / 100)
    epoch_list = list(np.arange(10, 30 + 1, 5))
    nodes_list = [[16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512]]

    # lowest dropout=0.1, because I know zero dropout will yield better training results but worse generalization (overfitting)
    dropouts = list(np.arange(1, 5) / 10)

    # earlier testing revealed relu performed significantly worse, so I removed it from the options
    activation_functions = ['tanh', 'sigmoid']
    batch_size_list = [32, 64, 128, 256, 512]

    ITERATIONS = 100
    results = pd.DataFrame(columns=['MSE', 'std_MSE',  # bigger std means less robust
                                    'alpha', 'epochs',
                                    'nodes', 'dropout',
                                    'activation', 'batch_size'])

    weights_file = 'mlp_hyper_parameter_weights.h5'  # save model weights
    specific_lags = [1, 2, 3, 4, 5, 10, 20]

    for i in range(ITERATIONS):
        if i < 10:
            print("Iteration ", str(i + 1))
        elif ((i + 1) % 10 == 0):
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
        df_train, train_label, _, idx = prep_data(df_train=train_org.drop(drop_sensors, axis=1),
                                                  train_label=train_org['RUL'],
                                                  df_test=x_test_org.drop(drop_sensors, axis=1),
                                                  remaining_sensors=remaining_sensors,
                                                  lags=specific_lags,
                                                  alpha=alpha)
        # create model
        input_dim = len(df_train.columns)
        model = create_model(input_dim, nodes_per_layer, dropout, activation, weights_file)

        # create train-validation split
        gss_search = GroupShuffleSplit(n_splits=3, train_size=0.80, random_state=42)
        for idx_train, idx_val in gss_search.split(df_train, train_label, groups=train_org.iloc[idx]['unit num']):
            X_train_split = df_train.iloc[idx_train].copy()
            y_train_split = train_label.iloc[idx_train].copy()
            X_val_split = df_train.iloc[idx_val].copy()
            y_val_split = train_label.iloc[idx_val].copy()

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

    results.to_csv("hyp_results", index=False)

alpha = 0.4
epochs = 30
specific_lags = [1, 2, 3, 4, 5, 10, 20]
nodes = [128, 256, 512]
dropout = 0.4
activation = 'sigmoid'
batch_size = 64

df_train, train_label, df_test, _ = prep_data(df_train=train_org.drop(drop_sensors, axis=1),
                                                  train_label=train_org['RUL'],
                                                  df_test=x_test_org.drop(drop_sensors, axis=1),
                                                  remaining_sensors=remaining_sensors,
                                                  lags=specific_lags,
                                                  alpha=alpha)

train = False
filename = 'finalized_tuned_NN_model.h5'
if train:
    input_dim = len(df_train.columns)
    weights_file = 'mlp_hyper_parameter_weights'
    final_model = create_model(input_dim,
                               nodes_per_layer=nodes,
                               dropout=dropout,
                               activation=activation,
                               weights_file=weights_file)

    final_model.compile(loss='mean_squared_error', optimizer='adam')
    final_model.load_weights(weights_file)

    final_model.fit(df_train, train_label, epochs=epochs, batch_size=batch_size, verbose=0)
    final_model.save(filename)  # save trained model

# predict and evaluate
final_model = load_model(filename)
print("tuned Neural Network")
y_hat_train = final_model.predict(df_train)
label, rmse, variance = evaluate(train_label, y_hat_train, 'train')
NN_train = ["NN (tuned)", label, rmse, variance]
list_results.append(NN_train)

y_hat_test = final_model.predict(df_test)
label, rmse, variance = evaluate(y_test, y_hat_test)
NN_test = ["NN (tuned)", label, rmse, variance]
list_results.append(NN_test)

################################
#   Random Survival Forest
################################

print(delimiter)
print("Started Random Survival Forest")

rsf_x = train_censored.copy()
rsf_x['RUL'] = rsf_x.RUL.astype('float')

rsf_y = rsf_x[['breakdown', 'cycle']]
rsf_y['breakdown'].replace(0, False, inplace=True)  # rsf only takes true or false
rsf_y['breakdown'].replace(1, True, inplace=True)  # rsf only takes true or false

attribute_dropped = ['unit num', 'cycle', 'RUL', 'breakdown', 'start']
rsf_x = rsf_x.drop(attribute_dropped, axis=1)
rsf_x_train, rsf_x_val, rsf_y_train, rsf_y_val = train_test_split(rsf_x, rsf_y, test_size=0.25)

random_state = 20
train = True
filename = 'finalized_rsf_model.sav'
if train:
    rsf = RandomSurvivalForest(n_estimators=10,
                               min_samples_split=100,
                               min_samples_leaf=150,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    print("Fitting rsf")
    rsf.fit(rsf_x_train, Surv.from_dataframe('breakdown', 'cycle', rsf_y_train))  # y only takes structured data
    pickle.dump(rsf, open(filename, 'wb'))  # save trained model

#######################################################################################################################
# from sklearn.preprocessing import OrdinalEncoder
# from sksurv.datasets import load_gbsg2
# from sksurv.preprocessing import OneHotEncoder
#
# X, y = load_gbsg2()
#
# grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
# grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)
#
# X_no_grade = X.drop("tgrade", axis=1)
# Xt = OneHotEncoder().fit_transform(X_no_grade)
# Xt = np.column_stack((Xt.values, grade_num))
#
# feature_names = X_no_grade.columns.tolist() + ["tgrade"]
#
# random_state = 20
# X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)
#
# rsf = RandomSurvivalForest(n_estimators=1000,
#                            min_samples_split=10,
#                            min_samples_leaf=15,
#                            max_features="sqrt",
#                            n_jobs=-1,
#                            random_state=random_state)
# rsf.fit(X_train, y_train)
#
# print("The rsf score for example is: ", rsf.score(X_test, y_test))
#
# a = np.empty(X_test.shape[0], dtype=[("age", float), ("pnodes", float)])
# a["age"] = X_test[:, 0]
# a["pnodes"] = X_test[:, 4]
#
# sort_idx = np.argsort(a, order=["pnodes", "age"])
# X_test_sel = pd.DataFrame(
#     X_test[np.concatenate((sort_idx[:3], sort_idx[-3:]))],
#     columns=feature_names)
# print(type(X_test_sel))
# print(X_test_sel)
# print(pd.Series(rsf.predict(X_test_sel)))
#######################################################################################################################

print("Predicting rsf")
rsf = pickle.load(open(filename, 'rb'))

# print(type(rsf_x_val.to_numpy()))
# print(type(Surv.from_dataframe('breakdown', 'RUL', rsf_y_val)))
# print(rsf_x_val.shape)
# print(rsf_x_val.to_numpy())
# print(rsf_x_val.to_numpy().shape)
# print(Surv.from_dataframe('breakdown', 'RUL', rsf_y_val).shape)
# print(Surv.from_dataframe('breakdown', 'RUL', rsf_y_val))
print(type(rsf_x_val))
print(rsf_x_val.iloc[0:2, :])

pred_risk = rsf.predict(rsf_x_val.iloc[0:2, :])
surv = rsf.predict_survival_function(rsf_x_val, return_array=True)  # return the survival function for each data point
print(rsf.score(rsf_x_val, Surv.from_dataframe('breakdown', 'cycle', rsf_y_val)))

rsf_predictions = []  # create a list to store the rmst of each survival curve

print(len(rsf_x_val))
for i, s in enumerate(surv):
    print(i)
    km_rmst = restricted_mean_survival_time(s, t=cut_off)  # calculate restricted mean survival time
    rsf_predictions.append(km_rmst)

label, rmse, variance = evaluate(rsf_y_val, rsf_predictions)
rsf_train = ["rsf (pre-tuned)", label, rmse, variance]
list_results.append(rsf_train)

################################
#   Save results of each model
################################

print(delimiter)
print("Saving all results")
df_results = pd.DataFrame(list_results, columns=results_header, mode='a', header=False)
df_results["timestamp"] = pd.to_datetime('today').normalize()
print(df_results)
df_results.to_csv("saved_results.csv")
