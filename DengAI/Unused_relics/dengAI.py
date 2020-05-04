# ProfitFB
# Early onset of outbreak


from __future__ import print_function
from __future__ import division
import torch

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

# just for the sake of this blog post!
from warnings import filterwarnings

from sk_models import sk_models

# -------------------------------PRE-PROCESSING------------------------
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
"""
print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)
"""


# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)

pd.isnull(sj_train_features).any()

(sj_train_features
 .ndvi_ne
 .plot
 .line(lw=0.8))

plt.title('Vegetation Index over Time')
plt.xlabel('Time')

sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)

"""
print('San Juan')
print('mean: ', sj_train_labels.mean()[0])
print('var :', sj_train_labels.var()[0])

print('\nIquitos')
print('mean: ', iq_train_labels.mean()[0])
print('var :', iq_train_labels.var()[0])
"""

sj_train_labels.hist()

iq_train_labels.hist()

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases

# compute the correlations
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()

# plot san juan
#sj_corr_heat = sns.heatmap(sj_correlations)
#plt.title('San Juan Variable Correlations')

# plot iquitos
#iq_corr_heat = sns.heatmap(iq_correlations)
#plt.title('Iquitos Variable Correlations')

# San Juan
(sj_correlations
 .total_cases
 .drop('total_cases')  # don't compare with myself
 .sort_values(ascending=False)
 .plot
 .barh())

# Iquitos
(iq_correlations
 .total_cases
 .drop('total_cases')  # don't compare with myself
 .sort_values(ascending=False)
 .plot
 .barh())


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']
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


# ---------------------------------MANUAL NEURAL NETWORK------------------------
from sklearn.model_selection import KFold

sj_train, iq_train = preprocess_data_all_features('dengue_features_train.csv', labels_path='dengue_labels_train.csv')

X_sj = torch.FloatTensor(sj_train.values)
y_sj = X_sj[:, -1]
X_sj = X_sj[:, :-1]
X_iq = torch.FloatTensor(iq_train.values)
y_iq = X_iq[:, -1]
X_iq = X_iq[:, :-1]


def scale1Darray(x):
    minimum = min(x)
    maximum = max(x)
    for a in x:
        a = (a - minimum) / (maximum - minimum)


# Scale columns
for column in X_sj.T:
    scale1Darray(column)
for column in X_iq.T:
    scale1Darray(column)

def train_manual_neural(m, criterion, optimizer, X, y, lr):
    epochs = 1000

    for epoch in range(epochs):

        # Forward pass
        y_pred = m(X)

        # Compute Loss
        loss = criterion(y_pred, y)

        m.zero_grad()

        #optimizer.zero_grad()
        # print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        with torch.no_grad():
            for param in m.parameters():
                param -= lr * param.grad


def eval_manual_neural(m, criterion, X, y):
    y_pred = m(X)
    after_train = criterion(y_pred, y)
    print('Evaluation Test loss ', after_train.item())
    return after_train.item()


def cross_val(m, criterion, optimizer, X, y, lr, k=8):
    kf = KFold(n_splits=k)
    scores = []
    for train_index, test_index in kf.split(X):
        print("Cross_VAL")
        #m = feed_forward(X.shape[1], 2)
        #model = TwoLayerNet(X_sj.shape[1], 2, 1)
        m = Feedforward(X_iq.shape[1], 10)  # TwoLayerNet(X_iq.shape[1], 10, 1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_manual_neural(m, criterion, optimizer, X_train, y_train,lr)
        score = eval_manual_neural(m, criterion, X_test, y_test)
        scores.append(score)

    return sum(scores) / len(scores)

def cross_val(m, criterion, optimizer, X, y, lr, hu, k=8):
    kf = KFold(n_splits=k)
    scores = []
    for train_index, test_index in kf.split(X):
        print("Cross_VAL")
        #m = feed_forward(X.shape[1], 2)
        #model = TwoLayerNet(X_sj.shape[1], 2, 1)
        m = Feedforward(X_iq.shape[1], hu)  # TwoLayerNet(X_iq.shape[1], 10, 1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_manual_neural(m, criterion, optimizer, X_train, y_train,lr)
        score = eval_manual_neural(m, criterion, X_test, y_test)
        scores.append(score)
    return sum(scores) / len(scores)


from unused_models.TwoLayerNet import TwoLayerNet
from FFN import Feedforward
# ---------------Initializing Model-------------
#model = feed_forward(X_sj.shape[1], 2)
#model = r_neural_net(X.shape[1], 1, 2, 2)
#criterion = torch.nn.L1Loss()
criterion = torch.nn.L1Loss()

model_sj = TwoLayerNet(X_sj.shape[1], 1, 1)
optimizer = torch.optim.SGD(model_sj.parameters(), lr=0.000000001)
train_manual_neural(model_sj, criterion, optimizer, X_sj, y_sj,0.005)

model_iq = Feedforward(X_iq.shape[1], 1)#TwoLayerNet(X_iq.shape[1], 10, 1)
old_params = {}
for name, params in model_iq.named_parameters():
    old_params[name] = params.clone()
#print(model_iq.state_dict())
train_manual_neural(model_iq, criterion, optimizer, X_iq, y_iq,0.05)
#print(model_iq.state_dict())
for name, params in model_iq.named_parameters():
    if (old_params[name] == params).all():
        print("True")

"""
scores = []
hu = []
lr = []
for c in range(0,5):
    for d in range(0,5):
        scores.append(cross_val(model_iq, criterion, optimizer, X_iq, y_iq, 0.1 ** c,4 ** d , 4))
        hu.append(4 ** d)
        lr.append(0.1 ** c)

print((scores, hu, lr))
"""

from keras.layers import Dense
from keras.models import Sequential
# Initialising the ANN
modelSJ = Sequential()

# Adding the input layer and the first hidden layer
modelSJ.add(Dense(32, activation = 'relu', input_dim = 20))

# Adding the second hidden layer
modelSJ.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
modelSJ.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer

modelSJ.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
modelSJ.compile(optimizer = 'SGD', loss = 'mean_absolute_error')

# Fitting the ANN to the Training set
modelSJ.fit(sj_train.iloc[:,:-1], sj_train.iloc[:,-1], batch_size = 10, epochs = 100)

# Initialising the ANN
modelIQ = Sequential()

# Adding the input layer and the first hidden layer
modelIQ.add(Dense(32, activation = 'relu', input_dim = 20))

# Adding the second hidden layer
modelIQ.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
modelIQ.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer

modelIQ.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
modelIQ.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
modelIQ.fit(sj_train.iloc[:,:-1], sj_train.iloc[:,-1], batch_size = 10, epochs = 200)


sj_test, iq_test = preprocess_data_all_features('dengue_features_test.csv')

y_pred_iq = modelIQ.predict(iq_test).astype('int')
y_pred_sj = modelSJ.predict(sj_test).astype('int')



# -----------------------------------SKLEARN------------------------------
def param_search(m, X, y, cross_val=2):
    best_score = -1000
    best_alpha = 0
    best_hl = 0
    best_maxiter = 0
    grid_alpha = np.arange(1e-07, 1e-05, 1e-06, dtype=np.float)
    grid_maxiter = np.arange(10, 200, 20, dtype=np.int)
    grid_hl = np.arange(1, 5, 1, dtype=np.int8)
    for alpha in grid_alpha:
        print(alpha)
        for maxiter in grid_maxiter:
            for hl in grid_hl:
                model = sk_models(8, cross_val, alpha, hl, maxiter)
                predictions = model.generate_predictions(X, y, X)
                score = model.cross_validate(X, y)
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
                    best_hl = hl
                    best_maxiter = maxiter
    return (best_score, best_alpha, best_hl, best_maxiter)


sj_train, iq_train = preprocess_data_all_features('dengue_features_train.csv', labels_path='dengue_labels_train.csv')

# PARAMETER TUNING
# params = param_search(8, sj_train.iloc[:,:-1], sj_train.iloc[:,-1])
# print(params)

# San Juan
print("-----MULTILAYER PERCEPTRON------- ")
model_sj_sklearn = sk_models(8, 10, 1e-05, 2)
# predictions = model.generate_predictions(sj_train.iloc[:,:-1], sj_train.iloc[:,-1], sj_train.iloc[:,:-1])
#score = model_sj_sklearn.cross_validate(sj_train.iloc[:, :-1], sj_train.iloc[:, -1])
#print(score)

# Iquitos
print("IQUITOS")
print("-----MULTILAYER PERCEPTRON------- ")
model_iq_sklearn = sk_models(8, 10, 1e-05, 2)
#score = model_iq_sklearn.cross_validate(iq_train.iloc[:, :-1], iq_train.iloc[:, -1])
#print(score)


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

sj_test, iq_test = preprocess_data_all_features('dengue_features_test.csv')
"""
model_sj = sk_models(7, 8)
model_iq = sk_models(7, 8)
sj_pred = model_sj.generate_predictions(sj_train.iloc[:, :-1], sj_train.iloc[:, -1], sj_test)
iq_pred = model_iq.generate_predictions(iq_train.iloc[:, :-1], iq_train.iloc[:, -1], iq_test)
"""

sj_test_tensor = torch.FloatTensor(sj_test.values)
iq_test_tensor = torch.FloatTensor(iq_test.values)

# Scale columns
for column in sj_test_tensor.T:
    scale1Darray(column)
for column in iq_test_tensor.T:
    scale1Darray(column)

sj_pred = model_sj(sj_test_tensor)
iq_pred = model_iq(iq_test_tensor)

submission = pd.read_csv("submission_format.csv",
                         index_col=[0, 1, 2])

#submission.total_cases = np.concatenate([sj_pred.data.numpy(), iq_pred.data.numpy()])
submission.total_cases = np.concatenate([y_pred_sj, y_pred_iq])
submission.to_csv("benchmark.csv")
#plt.show()

