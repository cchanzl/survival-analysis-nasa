import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import MinMaxScaler

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


def save_data_file(df, filename, FL=False):
    file_path = "Dataset/processed/"
    if FL:
        file_path = "FL_data/"
    filename = file_path + filename + ".csv"
    df.to_csv(filename, index=False)
    print("Saved ", filename, " successfully!")


def trend_extractor(df, sensor_names, L=20, K=20):
    other_headers = ['window num', 'unit num', 'RUL']
    new_sensor_features = []
    for n in sensor_names:
        n = n + "_mean"
        new_sensor_features.append(n)

    for n in sensor_names:
        n = n + "_trend"
        new_sensor_features.append(n)

    num_sens_feature = len(new_sensor_features)
    headers = [*new_sensor_features, 'window num', 'unit num', 'RUL']
    list_trend = [[] for _ in range(len(headers))]
    for engine in df['unit num'].unique():
        df_unit = df[df['unit num'] == engine]
        window_num = df_unit['window num'].unique().tolist()
        list_trend[num_sens_feature] = [*list_trend[num_sens_feature], *window_num]
        unit_num = [engine] * len(window_num)
        list_trend[num_sens_feature+1] = [*list_trend[num_sens_feature+1], *unit_num]
        for idx, sensor in enumerate(sensor_names):
            mean_list = []
            trend_list = []
            RUL_list = []
            for window in range(0, len(df_unit['window num'].unique())):
                offset = K * window

                mean = df_unit.iloc[offset:offset+L][sensor].mean()
                trend = np.polyfit(range(1, K+1), df_unit.iloc[offset:offset+L][sensor], 1)
                window_RUL = [df_unit.iloc[offset + L - 1]['RUL']]

                trend_list = [*trend_list, trend[0]]
                mean_list = [*mean_list, mean]
                RUL_list = [*RUL_list, *window_RUL]
            list_trend[idx] = [*list_trend[idx], *mean_list]
            list_trend[int(idx + num_sens_feature/2)] = [*list_trend[int(idx + num_sens_feature/2)], *trend_list]
            if idx == 0:
                list_trend[num_sens_feature + 1 + 1] = [*list_trend[num_sens_feature + 1 + 1], *RUL_list]

    df_trended = pd.DataFrame.from_records(map(list, zip(*list_trend)), columns=headers)
    return df_trended


def count_windows(length, L=20, K=20):
    count = 0
    while length >= L:
        length = length - K
        count += 1
    return count


def slicing_generator(df, sensor_names, L=20, K=20):
    other_headers = ['window num', 'unit num', 'RUL']
    headers = [*sensor_names, *other_headers]
    list_win = [[] for _ in range(len(headers))]  # sensor + window number + unit num + RUL
    for engine in df['unit num'].unique():
        df_unit = df[df['unit num'] == engine]
        #num_of_windows = np.math.floor((len(df_unit) - L + 1) / K)  # no of win is const across sensors of same engine
        num_of_windows = count_windows(len(df_unit), L, K)
        window_num = list(itertools.chain.from_iterable(itertools.repeat(x+1, K) for x in range(0, num_of_windows)))
        list_win[len(sensor_names)] = [*list_win[len(sensor_names)], *window_num]
        unit_num = [engine] * num_of_windows * K
        list_win[len(sensor_names)+1] = [*list_win[len(sensor_names)+1], *unit_num]
        for idx, sensor in enumerate(sensor_names):
            internal_list = []
            RUL_list = []
            for window in range(0, num_of_windows):
                offset = K * window
                window_slice = df_unit.iloc[offset:offset+L][sensor]
                if len(window_slice) < L:  # discard if ending number of instance in last window is smaller than K
                    break
                window_RUL = [df_unit.iloc[offset + L - 1]['RUL']] * K
                internal_list = [*internal_list, *window_slice]
                RUL_list = [*RUL_list, *window_RUL]
            list_win[idx] = [*list_win[idx], *internal_list]
            if idx == 0:
                list_win[len(sensor_names) + 1 + 1] = [*list_win[len(sensor_names) + 1 + 1], *RUL_list]

    df_win = pd.DataFrame.from_records(map(list, zip(*list_win)), columns=headers)
    return df_win


def polynomial_fitting(df, sensor_names, deg=3):
    # apply polynomial fitting
    df_poly = df.copy()
    for engine in df['unit num'].unique():
        df_unit = df[df['unit num'] == engine]
        for column in sensor_names:
            train_p = np.poly1d(np.polyfit(df_unit['cycle'], df_unit[column], deg))
            df_poly.loc[df_poly['unit num'] == engine, column] = train_p(range(0, len(df_unit[column])))
    return df_poly


def z_score_scaler(df_train, df_test, sensor_names):
    # copy the dataframe
    df_std_train = df_train.copy()
    df_std_test = df_test.copy()

    # apply the z-score method to each sensor data for each engine
    for i in df_std_train['unit num'].unique():
        df_train_unit = df_train[df_train['unit num'] == i]
        df_test_unit = df_test[df_test['unit num'] == i]
        for column in sensor_names:
            df_std_train.loc[df_std_train['unit num'] == i, column] = (df_train_unit[column] - df_train_unit[column].mean()) / df_train_unit[column].std()
            df_std_test.loc[df_std_test['unit num'] == i, column] = (df_test_unit[column] - df_test_unit[column].mean()) / df_test_unit[column].std()

    return df_std_train, df_std_test


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


def make_single_id(df):
    df["unit num"] = df["unit num"] * 1000
    df["unit num"] = df["unit num"] + df["cycle"]


def rename_cols(df):
    new_header = ["id", "y"]
    for i in range(0, len(df.columns) - len(new_header)):
        temp_name = "x"
        temp_name = temp_name + str(i)
        new_header.append(temp_name)
    cols = df.columns.tolist()
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    df = df[cols]
    df.columns = new_header
    return df


def fl_data_splitter(split, train, filename):
    # to create a single index identifying engine number and cycle
    make_single_id(train)

    # rename columns to fit FL format
    train.rename(columns={'RUL': 'y', 'unit num': 'id'}, inplace=True)

    # drop unused columns
    train.drop(labels=["cycle", "breakdown", "start"], axis=1, inplace=True)

    # rename remaining columns
    train = rename_cols(train)

    # split and save
    for i in range(0, len(split) + 1):
        if i == 0:  # first cut
            temp_df = train.loc[(train["id"] < (split[i] + 1) * 1000)]
        elif i == len(split):  # last cut
            temp_df = train.loc[(train["id"] >= (split[i - 1] + 1) * 1000)]
        else:  # middle splits
            temp_df = train.loc[(train["id"] >= (split[i]) * 1000) & (train["id"] < (split[i] + 1) * 1000)]
        save_data_file(temp_df, filename[i], True)


def file_name_generator(split_points, listname, listtype="train"):
    number_of_parties = len(split_points) + 1
    for i in range(0, number_of_parties):
        filename = "party_" + chr(65 + i) + "_" + listtype
        listname.append(filename)


################################
#   Data Settings
################################

# clip RUL if above a certain level to improve training
clip_level = 150

# set if pseudo right censoring be performed on dataset at designated cut-off
right_censoring = True
cut_off = 200

# Sensors selected based on MannKendall, sensor 2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20 and 21 are selected
drop_sensors = ['sens1', 'sens5', 'sens6', 'sens9', 'sens10', 'sens14',
                'sens16', 'sens18', 'sens19', 'sens22', 'sens23', 'sens24', 'sens25', 'sens26']

drop_labels = drop_sensors + ["op1", "op2", "op3"]

remaining_sensors = ['sens2', 'sens3', 'sens4', 'sens7', 'sens8', 'sens11',
                     'sens12', 'sens13', 'sens15', 'sens17', 'sens20', 'sens21']

all_sensors = drop_sensors + remaining_sensors

if __name__ == "__main__":

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

    ################################
    #   General Data pre-processing
    ################################

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

    # Save processed original data
    save_data_file(train_org, "train_org")
    save_data_file(test_org, "test_org")

    # apply a floor to RUL in training data. 125 is selected as the min cycle is 128. See EDA.
    train_clipped = train_org.copy()
    train_clipped['RUL'] = train_clipped['RUL'].clip(upper=clip_level)
    test_clipped = test_org.copy()
    test_clipped['RUL'] = test_clipped['RUL'].clip(upper=clip_level)

    # Apply psuedo right censoring in training data. Important to improve accuracy and simulate right censoring
    if right_censoring:
        train_clipped = train_clipped[train_clipped['cycle'] <= cut_off].copy()

    # Save processed original data
    save_data_file(train_clipped, "train_clipped")
    save_data_file(test_clipped, "test_clipped")

    ##########################################
    #   Data Preparation for RUL-RF
    ##########################################

    # https://ieeexplore.ieee.org/document/9281004/footnotes#footnotes
    # feature engineering performed in line with this paper

    rul_rf_train = train_clipped.copy()
    rul_rf_test = test_clipped.copy()

    # apply z-score normalisation
    rul_rf_train_std, rul_rf_test_std = z_score_scaler(rul_rf_train, rul_rf_test, remaining_sensors)

    # apply polynomial fitting
    rul_rf_train_std_poly = polynomial_fitting(rul_rf_train_std, remaining_sensors)
    rul_rf_test_std_poly = polynomial_fitting(rul_rf_test_std, remaining_sensors)

    # save_data_file(rul_rf_train_std_poly, "rul_rf_train_std_poly")
    # save_data_file(rul_rf_train_std_poly, "rul_rf_train_std_poly")

    # plot line graph
    # col = 'sens8'
    # unit = 1
    # plt.plot(rul_rf_train_std.loc[rul_rf_train_std['unit num'] == unit, 'cycle'],
    #          rul_rf_train_std.loc[rul_rf_train_std['unit num'] == unit, col])
    # plt.plot(rul_rf_train_std_poly.loc[rul_rf_train_std_poly['unit num'] == unit, 'cycle'],
    #          rul_rf_train_std_poly.loc[rul_rf_train_std_poly['unit num'] == unit, col])
    # plt.title(col)
    # plt.xlabel('cycle')
    # plt.ylabel('normalised sensor reading')
    # plt.show()

    # extracting features from rolling window
    rul_rf_train_win = slicing_generator(rul_rf_train_std_poly, remaining_sensors)
    rul_rf_test_win = slicing_generator(rul_rf_test_std_poly, remaining_sensors)
    # save_data_file(rul_rf_train_win, "rul_rf_train_win")

    # extract trend from extracted features
    rul_rf_train_trended = trend_extractor(rul_rf_train_win, remaining_sensors)
    rul_rf_test_trended = trend_extractor(rul_rf_test_win, remaining_sensors)

    save_data_file(rul_rf_train_trended, "rul_rf_train_trended")
    save_data_file(rul_rf_test_trended, "rul_rf_test_trended")

    ##########################################
    #   Data Preparation for FL Training
    ##########################################

    # This section is to prepare data for use in FATE Federated Learning package
    # Data used in FL must have the following headers
    # id   y   x0   x1   x2   x3   ...   xN
    # y must be the target label, in this case the RUL
    # See examples at https://github.com/FederatedAI/FATE/tree/master/examples/data

    train_split_points = [40, 80]
    test_split_points = [40, 80]
    no_of_parties = len(train_split_points) + 1  # number of parties in the FL

    train_file_names = []
    test_file_names = []

    file_name_generator(train_split_points, train_file_names, "train")
    file_name_generator(test_split_points, test_file_names, "test")

    minmax_scale = True
    train_no_rows = len(train_clipped)
    test_no_rows = len(test_clipped)

    # check that train and test are split into the same number
    if len(train_split_points) != len(test_split_points):
        sys.exit('Number of split points between train and test does not match')

    # check number of file names matches number of parties
    if len(train_file_names) != len(test_file_names) != no_of_parties * 2:
        sys.exit('Number of file names does not match number of parties')

    # Whether to apply minmax scaling
    if minmax_scale:
        train_clipped, test_clipped = minmax_scaler(train_clipped, test_clipped, remaining_sensors)

    # split and save files
    fl_data_splitter(train_split_points, train_clipped, train_file_names)
    fl_data_splitter(test_split_points, test_clipped, test_file_names)

