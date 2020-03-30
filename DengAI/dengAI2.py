# ProfitFB
# Early onset of outbreak


from __future__ import print_function
from __future__ import division
import torch
from feed_forward import feed_forward

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# just for the sake of this blog post!
from warnings import filterwarnings

from sk_models import sk_models
from sklearn.model_selection import KFold

filterwarnings('ignore')

# load the provided data
train_features = pd.read_csv('dengue_features_train.csv',
                             index_col=[0, 1, 2])

train_labels = pd.read_csv('dengue_labels_train.csv',
                           index_col=[0, 1, 2])

sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']

# Remove `week_start_date` string.
# sj_train_features.drop('week_start_date', axis=1, inplace=True)
# iq_train_features.drop('week_start_date', axis=1, inplace=True)

pd.isnull(sj_train_features).any()

(sj_train_features
 .ndvi_ne
 .plot
 .line(lw=0.8))

sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases


def preprocess_data_all_features(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    print(df.shape)
    # select features we want
    features = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
                'reanalysis_air_temp_k',
                'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent',
                'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c',
                'station_precip_mm']
    df = df[features]

    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj, iq


# Scaling input data
def scale1Darray(x):
    minimum = min(x)
    maximum = max(x)
    for a in x:
        a = (a - minimum) / (maximum - minimum)


sj_train, iq_train = preprocess_data_all_features('dengue_features_train.csv', labels_path='dengue_labels_train.csv')

from keras.layers import Dense, Activation
from keras.models import Sequential


# Sequential 4 layer neural network
def returnSequential4():
    model = Sequential()

    model.add(Dense(32, activation='relu', input_dim=20))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='Adam', loss='mean_absolute_error')

    return model


# Sequential 6 layer neural network
def returnSequential6():
    model = Sequential()
    model.add(Dense(50, input_dim=20, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='Adam', loss='mean_absolute_error')
    return model


def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_dim=20))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(optimizer='Adam', loss='mean_absolute_error')
    return model


def ensemble_nn():
    model1 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
    model2 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
    model3 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)

    ensemble_clf = VotingClassifier(estimators=[('model1', model1), ('model2', model2), ('model3', model3)],
                                    voting='soft')
    return ensemble_clf


def returnRandomForest(n_est=100):
    return RandomForestRegressor(n_estimators=n_est, criterion='mean_absolute_error', random_state=0)


def cross_val(bs, ep, X, y, k=3):
    print("CROSS VAL")
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    scores = []
    for train_index, test_index in kf.split(X):
        m = returnSequential6()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m.fit(X_train, y_train, batch_size=bs, epochs=ep, verbose=0)
        score = m.evaluate(X_test, y_test, verbose=0)
        scores.append(score)

    return sum(scores) / len(scores)


def returnOptimizedModel(X, y):
    # SJ Optimization
    criterion = torch.nn.MSELoss()
    scores = []
    aa = []
    bb = []
    for a in range(10, 100, 20):
        for b in range(100, 501, 200):
            scores.append(cross_val(a, b, X, y))
            aa.append(a)
            bb.append(b)

    print("MIN SCORE: ", min(scores), " \n")
    index = scores.index(min(scores))
    print("Batch: ", aa[index], " epochs: ", bb[index])
    model = returnSequential6()
    model.fit(X, y, batch_size=aa[index], epochs=bb[index], verbose=0)
    return model


X_sj = sj_train.iloc[:, :-1]
y_sj = sj_train.iloc[:, -1]
X_iq = iq_train.iloc[:, :-1]
y_iq = iq_train.iloc[:, -1]

from sklearn.preprocessing import scale

X_sj = scale(X_sj, axis=1, with_mean=True, with_std=True)
X_iq = scale(X_iq, axis=1, with_mean=True, with_std=True)

model_sj = returnOptimizedModel(X_sj, y_sj)
model_iq = returnOptimizedModel(X_iq, y_iq)

model_sj.fit(X_sj, y_sj)
model_iq.fit(X_iq, y_iq)

# SUBMISSION
sj_test, iq_test = preprocess_data_all_features('dengue_features_test.csv')

from sklearn.preprocessing import scale

sj_test = scale(sj_test.iloc[:, :], axis=1, with_mean=True, with_std=True)
iq_test = scale(iq_test.iloc[:, :], axis=1, with_mean=True, with_std=True)

y_pred_sj = model_sj.predict(sj_test).astype('int')
y_pred_iq = model_iq.predict(iq_test).astype('int')
submission = pd.read_csv("submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([y_pred_sj, y_pred_iq])
submission.to_csv("submission.csv")


# Go through all SKLEARN models
def sklearn_models():
    for x in range(9):
        predictions_sj = []
        predictions_iq = []
        if (x != 6):
            print("X ", x)
            model = sk_models(x, 4)
            sj_test, iq_test = preprocess_data_all_features('dengue_features_test.csv')

            prediction_sj = model.generate_predictions(sj_train.iloc[:, :-1], sj_train.iloc[:, -1],
                                                       sj_test)
            prediction_iq = model.generate_predictions(iq_train.iloc[:, :-1], iq_train.iloc[:, -1],
                                                       iq_test)
            print("SJ")
            model.cross_validate(sj_train.iloc[:, :-1], sj_train.iloc[:, -1])
            print("IQ")
            model.cross_validate(iq_train.iloc[:, :-1], iq_train.iloc[:, -1])
            predictions_sj.append(prediction_sj)
            predictions_iq.append(prediction_iq)

# With shuffling, both net6 and net4 hover around 30 MAE testing.
