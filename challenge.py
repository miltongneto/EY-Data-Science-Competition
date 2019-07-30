#!/usr/bin/env python
# coding: utf-8

# ## EY Data Science Challenge

import pandas as pd
import numpy as np
import geopandas as gpd
import datetime
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost.sklearn import XGBClassifier

def load_data(filename):
    data = pd.read_csv('data/raw/' + filename + '.csv', index_col=0)
    data['time_entry'] = pd.to_datetime(data['time_entry'], format='%H:%M:%S')
    data['time_exit'] = pd.to_datetime(data['time_exit'], format='%H:%M:%S')

    data['hash'] = data['hash'].astype('category')
    data['hash'] = data['hash'].astype('category')
    print('Shape: ', data.shape)

    return data


# ### Create new variables


def create_variable_is_center(data):
    # For ENTRY
    condition_x = ((data['x_entry'] >= 3750901.5068) & (data['x_entry'] <= 3770901.5068))
    condition_y = ((data['y_entry'] >= -19268905.6133) & (data['y_entry'] <= -19208905.6133))

    data['is_center_entry'] = 0
    data.loc[condition_x & condition_y, 'is_center_entry'] = 1

    # For EXIT
    condition_x = ((data['x_exit'] >= 3750901.5068) & (data['x_exit'] <= 3770901.5068))
    condition_y = ((data['y_exit'] >= -19268905.6133) & (data['y_exit'] <= -19208905.6133))

    data['is_center_exit'] = 0
    data.loc[condition_x & condition_y, 'is_center_exit'] = 1

    return data


def create_points(data):
    data['point_entry'] = data.apply(lambda x: Point(x['x_entry'], x['y_entry']), axis=1)
    data['point_exit'] = data.apply(lambda x: Point(x['x_exit'], x['y_exit']), axis=1)
    return data


def delete_points(data):
    return data.drop(['point_entry', 'point_exit'], axis=1)


# #### Distance between entry and exit
def distance_trajectory(data):
    data['distance'] = data.apply(lambda x: x['point_entry'].distance(x['point_exit']), axis=1)
    return data


# #### Duration of the trajectory  and average speed
# average speed using distance computed from the points
def duration_trajectory(data):
    data['duration'] = data['time_exit'] - data['time_entry']
    data['duration'] = data['duration'].dt.total_seconds() / 60

    data['my_vmean'] = data['distance'] / data['duration']
    data['my_vmean'].fillna(0, inplace=True)

    return data


# #### Time features
# Hour, minute and part of the day
def create_time_features(data):
    data['time_entry_hour'] = data['time_entry'].dt.hour
    data['time_entry_minute'] = data['time_entry'].dt.minute
    
    data['time_exit_hour'] = data['time_exit'].dt.hour
    data['time_exit_minute'] = data['time_exit'].dt.minute
    
    data['time_entry_part_day'] = pd.cut(data['time_entry'].dt.hour, [0, 4, 8 , 12, 16],
                                         labels=['night', 'early morning', 'morning', 'afternoon'], include_lowest=True)
    data['time_exit_part_day'] = pd.cut(data['time_exit'].dt.hour, [0, 4, 8 , 12, 16],
                                        labels=['night', 'early morning', 'morning', 'afternoon'], include_lowest=True)
    return data


# #### Distance between the center and entry
def distance_center(data):
    center = Polygon([(3750901.5068, -19268905.6133), (3750901.5068, -19208905.6133),
                      (3770901.5068, -19208905.6133), (3770901.5068, -19268905.6133)])

    data['distance_center'] = data.apply(lambda x: x['point_entry'].distance(center), axis=1)
    data['distance_center_exit'] = data.apply(lambda x: x['point_exit'].distance(center), axis=1)

    data['distance_boundary_center'] = data.apply(lambda x: x['point_entry'].distance(center.boundary), axis=1)
    data['distance_boundary_center_exit'] = data.apply(lambda x: x['point_exit'].distance(center.boundary), axis=1)

    x_c = (3770901.5068 + 3750901.5068) / 2
    y_c = (-19208905.6133 - 19268905.6133) / 2
    center_point = Point(x_c, y_c)

    data['distance_center_point'] = data.apply(lambda x: x['point_entry'].distance(center_point), axis=1)
    data['distance_center_point_exit'] = data.apply(lambda x: x['point_exit'].distance(center_point), axis=1)

    data['approach_center_point'] = data['distance_center_point'] - data['distance_center_point_exit']
    data['approach_center'] = data['distance_boundary_center'] - data['distance_boundary_center_exit']

    return data


# ### Separate train and test set
def separate_train_test(data):
    after_15 = data[data['time_exit'].dt.time >= datetime.time(hour=15, minute=0)]
    #hashes = df_train.groupby('hash').size().index.categories

    train, test = train_test_split(after_15, test_size=0.3)
    train = data[data['hash'].isin(train['hash'])]
    test = data[data['hash'].isin(test['hash'])]

    return train, test


# ### Pre-processing
def categorize_features(data):
    entry_part_day = pd.get_dummies(data['time_entry_part_day'])
    entry_part_day.columns = [i + '_entry' for i in entry_part_day.columns]

    exit_part_day = pd.get_dummies(data['time_exit_part_day'])
    exit_part_day.columns = [i + '_exit' for i in exit_part_day.columns]

    data = pd.concat([data, entry_part_day, exit_part_day], axis=1)
    data.drop(['time_entry_part_day', 'time_exit_part_day'], axis=1, inplace=True)

    return data


def rename_column_name(trajectory, count, trajectories):
    trajectory.index = trajectory.index.map(lambda x: str(x) + '_' + str(count[0]))
    trajectories.append(trajectory)
    count[0] = count[0] - 1


def create_dataset(data):
    k = 20
    rows = []
    #trajectories_ids = []
    count = 1
    size = len(data['hash'].unique())
    for h in data['hash'].unique():
        if count % 500 == 0:
            print('{}/{}'.format(count, size))

        df_user = data[data['hash'] == h]
        trajectory_id = df_user.iloc[-1]['trajectory_id']
        df_user = df_user.drop(['hash', 'trajectory_id'], axis=1)

        last_trajectory = df_user.iloc[[-1]].reset_index(drop=True)

        target = last_trajectory['is_center_exit']
        last_trajectory = last_trajectory.drop(
            [
                'vmax', 'vmin', 'vmean', 'x_exit', 'y_exit', 'distance', 'is_center_exit', 'my_vmean',
                'distance_center_exit', 'distance_boundary_center_exit', 'distance_center_point_exit',
                'approach_center_point', 'approach_center'
            ], axis=1)
        last_trajectory.columns = last_trajectory.columns.map(lambda x: str(x) + '_last')

        last_trajectory['distance_mean'] = df_user.iloc[:-1]['distance'].mean()
        last_trajectory['duration_mean'] = df_user.iloc[:-1]['duration'].mean()
        last_trajectory['my_vmean_mean'] = df_user.iloc[:-1]['my_vmean'].mean()
        last_trajectory['distance_estimated'] = last_trajectory['my_vmean_mean'] * last_trajectory['duration_mean']
        last_trajectory['amount_trajectories'] = df_user.shape[0]

        # FOR FEATURES IN DESCENDING ORDER

        trajectories = [last_trajectory]

        amount_trajectories = df_user.shape[0] - 1 - 1   # 1 by the last element and 1 by consider the zero
        j = k - 1
        for i in range(amount_trajectories, max(amount_trajectories - k + 1, -1), -1):
            aux = df_user.iloc[[i]]
            aux.columns = aux.columns.map(lambda x: str(x) + '_' + str(j))
            aux.reset_index(drop=True, inplace=True)
            trajectories.append(aux)
            j -= 1

        row = pd.concat(trajectories, axis=1, ignore_index=False)

        row['target'] = target
        row['trajectory_id'] = trajectory_id
        rows.append(row)

        count += 1

    dataset = pd.concat(rows, sort=False)

    return dataset


def remove_columns(data):
    data.drop(['time_entry', 'time_exit'], axis=1, inplace=True)
    return data


def clean_data(data):
    data.loc[:, data.columns != 'hash'] = data.loc[:, data.columns != 'hash'].fillna(0)
    data.reset_index(drop=True, inplace=True)
    data.replace(np.inf, 0, inplace=True)
    return data


def create_direction_variables(data):
    vertical = pd.get_dummies(np.sign(data['y_exit'] - data['y_entry']))
    horizontal = pd.get_dummies(np.sign(data['x_exit'] - data['x_entry']))
    vertical.columns = ['vertical_down', 'vertical_constant', 'vertical_up']
    horizontal.columns = ['horizontal_left', 'horizontal_constant', 'horizontal_right']

    data = pd.concat([data, vertical, horizontal], axis=1)
    return data


def pre_process(data, data_type):
    print("Pre-processing")
    data = create_points(data)
    data = create_variable_is_center(data)
    data = distance_trajectory(data)
    data = duration_trajectory(data)
    data = distance_center(data)
    data = create_time_features(data)
    # data = create_direction_variables(data)
    data = categorize_features(data)
    data = delete_points(data)

    return data


def separate_features_target_id(data, cols=None):
    if cols:
        X = data.loc[:, cols]
    else:
        X = data.drop(['target', 'trajectory_id'], axis=1)

    y = data.loc[:, 'target'].copy()

    trajectory_id = data.loc[:, 'trajectory_id'].copy()

    return X, y, trajectory_id


def train_evaluate(X_train, y_train, X_test, y_test):
    print('Training model')
    model = XGBClassifier(n_jobs=-1, n_estimators=1000, learning_rate=0.05, max_depth=6, 
        colsample_bytree=0.9, subsample=0.8, min_child_weight=4, reg_alpha=0.005)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print('Results:')
    print(accuracy_score(y_test, pred))
    print(f1_score(y_test, pred))

    return pred


def main(mode):
    train = load_data('data_train')
    if mode == 'train':
        train, test = separate_train_test(train)
    else:
        test = load_data('data_test')

    train = pre_process(train, 'train')
    train = remove_columns(train)
    train_set = create_dataset(train)
    train_set = clean_data(train_set)
    train_set.to_csv('data/processed/train_dataset_processed.csv', index=False)

    test = pre_process(test, 'test')
    test = remove_columns(test)
    test_set = create_dataset(test)
    test_set = clean_data(test_set)
    test_set.to_csv('data/processed//test_dataset_processed.csv', index=False)

    X_train, y_train, id_train = separate_features_target_id(train_set, cols=None)
    X_test, y_test, id_test = separate_features_target_id(test_set, cols=X_train.columns.to_list())

    pred = train_evaluate(X_train, y_train, X_test, y_test)
    result = pd.concat([id_test, pd.Series(pred)], axis=1)
    result.to_csv('result.csv', index=False)


if __name__ == '__main__':
    main('train')

