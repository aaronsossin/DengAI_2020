# ProfitFB
# Early onset of outbreak


from __future__ import print_function
from __future__ import division
import torch
from feed_forward import feed_forward
from sklearn.preprocessing import scale
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt

from statistics import stdev, mean
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


def convertToEpidemic(y):
    m = mean(y)
    std = stdev(y)
    threshold = m + 2.0 * std
    newY = [1 if x > threshold else 0 for x in y]
    return np.array(newY)


def numberEpidemics(y):
    counter = 0
    indices = []
    lastOne = False
    for a in range(len(y)):
        if y[a] == 1 and (not lastOne):
            counter = counter + 1
            indices.append(a)
            lastOne = True
        elif y[a] == 0:
            lastOne = False

    print(counter)
    print(indices)
    return counter


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


from keras.layers import SimpleRNN


def RNN():
    model = Sequential()
    model.add(SimpleRNN(2, input_dim=20))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='Adam', loss='mean_absolute_error')
    return model


def multi_RNN():
    model = Sequential()
    model.add(SimpleRNN(2, input_dim=20))
    model.add(SimpleRNN(2, input_dim=20))
    model.add(SimpleRNN(2, input_dim=20))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='Adam', loss='mean_absolute_error')
    return model


def mlp_model():
    model = Sequential()

    model.add(Dense(50, input_dim=20))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.compile(optimizer='Adam', loss='mean_absolute_error')
    return model


from keras.layers import LSTM


def ensemble_nn():
    model1 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
    model2 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
    model3 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)

    ensemble_clf = VotingClassifier(estimators=[('model1', model1), ('model2', model2), ('model3', model3)],
                                    voting='soft')
    return ensemble_clf


def lstm():
    model = Sequential()
    model.add(LSTM(4, input_dim=20))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


def multi_lstm():
    model = Sequential()
    model.add(LSTM(4, input_dim=20, return_sequences=True))
    model.add(LSTM(4, input_dim=20))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


def returnRandomForest(n_est=100):
    return RandomForestRegressor(n_estimators=n_est, criterion='mean_absolute_error', random_state=0)


def cross_val_recurrent(bs, ep, X, y, k=3):
    print("CROSS VAL")
    kf = KFold(n_splits=k, shuffle=False, random_state=0)
    scores = []
    for train_index, test_index in kf.split(X):
        m = RNN()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        m.fit(X_train, y_train, batch_size=bs, epochs=ep, verbose=0)
        # m.fit(X_train, y_train)
        score = m.evaluate(X_test, y_test, verbose=0)
        scores.append(score)

    return sum(scores) / len(scores)

def cross_val(bs, ep, X, y, k=5):
    print("CROSS VAL")
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    scores = []
    for train_index, test_index in kf.split(X):
        m = baseline()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m.fit(X_train, y_train, batch_size=bs, epochs=ep, verbose=0)
        # m.fit(X_train, y_train)
        a,score = m.evaluate(X_test, y_test, verbose=0)
        scores.append(score)

    return sum(scores) / len(scores)
import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

from keras.metrics import Precision, TruePositives
def baseline():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[TruePositives()])
    return model


def returnOptimizedModel(X, y):
    # SJ Optimization
    criterion = torch.nn.MSELoss()
    scores = []
    aa = []
    bb = []
    for a in range(10, 100, 20):  # 10, 100, 20
        for b in range(100, 501, 200):  # 100, 501, 200
            scores.append(cross_val(a, b, X, y))
            aa.append(a)
            bb.append(b)

    print("MIN SCORE: ", min(scores), " \n")
    index = scores.index(min(scores))
    print("MIN SCORE: ", min(scores))
    print("Batch: ", aa[index], " epochs: ", bb[index])
    model = baseline()
    model.fit(X, y, batch_size=aa[index], epochs=bb[index], verbose=0)
    # model.fit(X, y)
    return model

def returnOptimizedModel_binary(X, y):
    # SJ Optimization
    criterion = torch.nn.MSELoss()
    scores = []
    aa = []
    bb = []
    for a in range(10, 100, 20):  # 10, 100, 20
        for b in range(100, 501, 200):  # 100, 501, 200
            scores.append(cross_val(a, b, X, y))
            aa.append(a)
            bb.append(b)

    print("MAX SCORE: ", max(scores), " \n")
    index = scores.index(max(scores))
    print("Max SCORE: ", max(scores))
    print("Batch: ", aa[index], " epochs: ", bb[index])
    model = baseline()
    model.fit(X, y, batch_size=aa[index], epochs=bb[index], verbose=0)
    # model.fit(X, y)
    return model


def newIdea(X, y):
    values = (y.tolist()).copy()
    values.sort()
    a = values[int(len(values) / 3)]
    b = values[int(len(values) * 2 / 3)]
    addFeature = []
    for z in range(len(values)):
        if values[z] <= a:
            addFeature.append(1)
        if values[z] >= a:
            addFeature.append(3)
        else:
            addFeature.append(2)
    return addFeature


X_sj = sj_train.iloc[:, :-1]
y_sj = sj_train.iloc[:, -1]
X_iq = iq_train.iloc[:, :-1]
y_iq = iq_train.iloc[:, -1]

X_sj = scale(X_sj, axis=1, with_mean=True, with_std=True)
X_iq = scale(X_iq, axis=1, with_mean=True, with_std=True)

y_sj = convertToEpidemic(y_sj)
y_iq = convertToEpidemic(y_iq)
"""
values = (y_sj.tolist()).copy()
values.sort()
print(values)
sjadded = [1 if x < values[int(len(values)/3)] else 3 if x < values[int(len(values)/3)] else 2 for x in values]
values = y_iq.tolist().copy()
values.sort()
iqadded = [1 if x < values[int(len(values)/3)] else 3 if x < values[int(len(values)/3)] else 2 for x in values]
X_sj = np.hstack((X_sj, sjadded))
X_iq = np.hstack((X_iq, iqadded))
"""

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
# X_sj = poly.fit_transform(X_sj)
# X_iq = poly.fit_transform(X_iq)

#model_sj = returnOptimizedModel_binary(X_sj, y_sj)
#odel_iq = returnOptimizedModel_binary(X_iq, y_iq)
model_sj = baseline()
model_sj.fit(X_sj, y_sj, epochs=5000, verbose=0)
print(model_sj.evaluate(X_sj,y_sj))
model_iq = baseline()
model_iq.fit(X_iq, y_iq, epochs=1000, verbose=0)
#X_sj = np.reshape(X_sj, (X_sj.shape[0], 1, X_sj.shape[1]))
#X_iq = np.reshape(X_iq, (X_iq.shape[0], 1, X_iq.shape[1]))

fig1, ax1 = plt.subplots()

sj_pred = model_sj.predict(X_sj)
sj_pre = np.array([1 if x >= 0.5 else 0 for x in sj_pred])
print(sum(sj_pre))
iq_pred = model_iq.predict(X_iq)
iq_pre = np.array([1 if x >= 0.5 else 0 for x in iq_pred])
print(sum(iq_pre))


# plot sj
plt.plot(range(len(y_sj)), sj_pre)
plt.plot(range(len(y_sj)), y_sj)

fig2, ax2 = plt.subplots()
plt.plot(range(len(y_iq)), iq_pre)
plt.plot(range(len(y_iq)), y_iq)
plt.show()
# SUBMISSION
sj_test, iq_test = preprocess_data_all_features('dengue_features_test.csv')

sj_test = scale(sj_test.iloc[:, :], axis=1, with_mean=True, with_std=True)
iq_test = scale(iq_test.iloc[:, :], axis=1, with_mean=True, with_std=True)

#sj_test = np.reshape(sj_test, (sj_test.shape[0], 1, sj_test.shape[1]))
#iq_test = np.reshape(iq_test, (iq_test.shape[0], 1, iq_test.shape[1]))

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
# sj_test = poly.fit_transform(sj_test)
# iq_test = poly.fit_transform(iq_test)

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
