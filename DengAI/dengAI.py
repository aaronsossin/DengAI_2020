# ProfitFB
# Early onset of outbreak


from __future__ import print_function
from __future__ import division
from RNN import r_neural_net
import torch
from feed_forward import feed_forward
from neural_net_1 import neural_net_1

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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

print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)

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

print('San Juan')
print('mean: ', sj_train_labels.mean()[0])
print('var :', sj_train_labels.var()[0])

print('\nIquitos')
print('mean: ', iq_train_labels.mean()[0])
print('var :', iq_train_labels.var()[0])

sj_train_labels.hist()

iq_train_labels.hist()

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases

# compute the correlations
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()

# plot san juan
sj_corr_heat = sns.heatmap(sj_correlations)
plt.title('San Juan Variable Correlations')

# plot iquitos
iq_corr_heat = sns.heatmap(iq_correlations)
plt.title('Iquitos Variable Correlations')

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
    features = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
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


sj_train, iq_train = preprocess_data_all_features('dengue_features_train.csv',
                                                  labels_path="dengue_labels_train.csv")

sj_test, iq_test = preprocess_data_all_features('dengue_features_test.csv')

# ---------------------------------MANUAL NEURAL NETWORK------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

X = torch.FloatTensor(sj_train.values)
X_test = torch.FloatTensor(sj_test.values)
y = X[:, -1]
X = X[:, :-1]


def scale1Darray(x):
    minimum = min(x)
    maximum = max(x)
    for a in x:
        a = (a - minimum) / (maximum- minimum)

#Scale columns
for column in X.T:
    scale1Darray(column)

print(X)


def train_manual_neural(m, criterion, optimizer, X, y):
    epochs = 5000
    for epoch in range(epochs):
        m.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X)

        # Compute Loss
        loss = criterion(y_pred, y)

        # print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()


def eval_manual_neural(m, criterion, X, y):
    y_pred = m(X)
    after_train = criterion(y_pred, y)
    print('Test loss ', after_train.item())
    return after_train.item()


def cross_val(m, criterion, optimizer, X, y, k=8):
    kf = KFold(n_splits=k)
    scores = []
    for train_index, test_index in kf.split(X):
        m = feed_forward(X.shape[1], 2)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_manual_neural(m, criterion, optimizer, X_train, y_train)
        score = eval_manual_neural(m, criterion, X_test, y_test)
        scores.append(score)

    return sum(scores) / len(scores)


# ---------------Initializing Model-------------
model = feed_forward(X.shape[1], 2)

# model = r_neural_net(X.shape[1], 1, 2, 2)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

eval_manual_neural(model, criterion, X, y)
print("Average score manual neural: ", cross_val(model, criterion, optimizer, X, y))


# train_manual_neural(model, criterion, optimizer, X, y)

# eval_manual_neural(model, criterion, X, y)

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


# PARAMETER TUNING
# params = param_search(8, sj_train.iloc[:,:-1], sj_train.iloc[:,-1])
# print(params)

# San Juan
print("SAN JUAN")
print("-----MULTILAYER PERCEPTRON------- ")
model_sj = sk_models(8, 10, 1e-05, 2)
# predictions = model.generate_predictions(sj_train.iloc[:,:-1], sj_train.iloc[:,-1], sj_train.iloc[:,:-1])
score = model_sj.cross_validate(sj_train.iloc[:, :-1], sj_train.iloc[:, -1])
print(score)

# Iquitos
print("IQUITOS")
print("-----MULTILAYER PERCEPTRON------- ")
model_iq = sk_models(8, 10, 1e-05, 2)
score = model_iq.cross_validate(iq_train.iloc[:, :-1], iq_train.iloc[:, -1])
print(score)

# print(score)
""" 
model = sk_models(1,10)
predictions = model.generate_predictions(sj_train.iloc[:800,:-1], sj_train.iloc[:800,-1], sj_train.iloc[:,:-1])
#print(predictions)
score = model.cross_validate(sj_train.iloc[:,:-1], sj_train.iloc[:,-1])
#print(score)




fig10, ax10 = plt.subplots()
ax10.plot(predictions)
ax10.plot(y)
ax10.set_title("Axis 1 title")
ax10.set_xlabel("X-label for axis 1")
plt.show()
"""
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

""""
def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c"

    grid = 10 ** np.arange(-8, -3, dtype=np.float64)

    best_alpha = []
    best_score = 1000

    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)
        score = eval_measures.mse(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)

    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model


sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)

figs, axes = plt.subplots(nrows=2, ncols=1)

# plot sj
sj_train['fitted'] = sj_best_model.fittedvalues
sj_train.fitted.plot(ax=axes[0], label="Predictions")
sj_train.total_cases.plot(ax=axes[0], label="Actual")

# plot iq
iq_train['fitted'] = iq_best_model.fittedvalues
iq_train.fitted.plot(ax=axes[1], label="Predictions")
iq_train.total_cases.plot(ax=axes[1], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
"""
sj_test, iq_test = preprocess_data_all_features('dengue_features_test.csv')

# sj_predictions = sj_best_model.predict(sj_test).astype(int)
# iq_predictions = iq_best_model.predict(iq_test).astype(int)
model_sj = sk_models(7, 8)
model_iq = sk_models(7, 8)
sj_pred = model_sj.generate_predictions(sj_train.iloc[:, :-1], sj_train.iloc[:, -1], sj_test)
iq_pred = model_iq.generate_predictions(iq_train.iloc[:, :-1], iq_train.iloc[:, -1], iq_test)

submission = pd.read_csv("submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_pred, iq_pred])
submission.to_csv("benchmark.csv")
plt.show()
