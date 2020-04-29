import Preprocess
from DeepModels import DeepModels
from pygam import LinearGAM
import numpy as np
import pandas as pd
import Epidemic
from EpidemicModels import EpidemicModels
from matplotlib import pyplot as plt

def returnOptimizedModel_epidemic(X, y, bs=0, ep=0):
    # SJ Optimization
    scores = []
    aa = []
    bb = []
    if bs == 0:
        for a in range(10, 100, 20):  # 10, 100, 20
            for b in range(100, 501, 200):  # 100, 501, 200
                scores.append(EpidemicModels(e).cross_eval(X,y,a,b,5))
                aa.append(a)
                bb.append(b)
    else:
        aa = [bs]
        bb = [ep]
        scores.append(100)
    print("MAX SCORE: ", max(scores), " \n")
    max_score = max(scores)
    index = scores.index(max_score)
    batch = aa[index]
    epochs = bb[index]
    model = EpidemicModels(e)
    model.train(X, y, batch, epochs)
    # model.fit(X, y)
    return model, max_score, batch, epochs

def returnOptimizedModel(X, y, idim=20):
    scores = []
    aa = []
    bb = []

    for a in range(10, 100, 20):  # 10, 100, 20
        for b in range(100, 501, 200):  # 100, 501, 200
            print("a: ", a, " b: ", b)
            scores.append(DeepModels(m,idim).cross_eval(X,y,a,b,5))
            aa.append(a)
            bb.append(b)

    min_score = min(scores)
    index = scores.index(min(scores))
    batch = aa[index]
    epochs = bb[index]
    optimized_model = DeepModels(m, idim)
    optimized_model.train(X, y, batch, epochs)
    return optimized_model, min_score, batch, epochs

import torch

X_sj, X_iq, y_sj, y_iq = Preprocess.extract()
sj_pre=[]
model_sj = DeepModels(0) #Placeholder
model_iq = DeepModels(0) #Placeholder
# Models
# 0. baseline (2-layer sequential dense layers)
# 1 4-layer sequential
# 2 6-layer sequential
#
combine_epidemic = True
epidemic = False
e = 1
m = 6


if combine_epidemic:
    y_sje = Epidemic.convertToEpidemic(y_sj)
    y_iqe = Epidemic.convertToEpidemic(y_iq)
    model_sj, max_score, batch, epochs = returnOptimizedModel_epidemic(X_sj, y_sj, 90, 100)
    model_iq, max_score, batch, epochs = returnOptimizedModel_epidemic(X_iq, y_iq, 90, 100)
    feature_e_sj = model_sj.prediction(X_sj)
    feature_e_iq = model_iq.prediction(X_iq)
    X_sj = np.hstack((X_sj, feature_e_sj))
    X_iq = np.hstack((X_iq, feature_e_iq))
    model_sj, min_score, batch, epochs = returnOptimizedModel(X_sj, y_sj, 21)
    print(min_score, " ", batch, " ", epochs)
    model_iq, min_score, batch, epochs = returnOptimizedModel(X_iq, y_iq, 21)
    print(min_score, " ", batch, " ", epochs)
elif epidemic:
    y_sj = Epidemic.convertToEpidemic(y_sj)
    y_iq = Epidemic.convertToEpidemic(y_iq)
    if m == 7:
        model_sj = EpidemicModels(m)
        model_sj.train(X_sj, y_sj)
        print("MAE: ", model_sj.cross_eval(X_sj, y_sj))
    else:
        model_sj, min_score, batch, epochs = returnOptimizedModel_epidemic(X_sj, y_sj, 90, 100)
        print(min_score, " ", batch, " ", epochs)
        #model_iq, min_score, batch, epochs = returnOptimizedModel_epidemic(X_iq, y_iq)
else:
    if m == 7:
        model_sj = DeepModels(m)
        model_sj.train(X_sj, y_sj)
        print("MAE: ", model_sj.cross_eval(X_sj, y_sj))
    else:
        model_sj, min_score, batch, epochs = returnOptimizedModel(X_sj, y_sj)
        print(min_score, " ", batch, " ", epochs)
        model_iq, min_score, batch, epochs = returnOptimizedModel(X_iq, y_iq)
        print(min_score, " ", batch, " ", epochs)

sj_pre = np.round(model_sj.prediction(X_sj))
fig1, ax1 = plt.subplots()
# plot sj
plt.plot(range(len(y_sj)), sj_pre, 'm', alpha=0.4)
plt.plot(range(len(y_sj)), y_sj, 'g')
plt.show()

sj_test, iq_test = Preprocess.preprocess_data_all_features('dengue_features_test.csv')
y_pred_sj = model_sj.prediction(sj_test).astype('int')
y_pred_iq = model_iq.prediction(iq_test).astype('int')

submission = pd.read_csv("submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([y_pred_sj, y_pred_iq])
submission.to_csv("submission.csv")


#Effect of scaling:
#Effect of differnt things:
#Effect of shuffling:
#Takeaways


