import Preprocess
from DeepModels import DeepModels
from pygam import LinearGAM
def returnOptimizedModel(X, y):
    scores = []
    aa = []
    bb = []

    for a in range(10, 100, 20):  # 10, 100, 20
        for b in range(100, 501, 200):  # 100, 501, 200
            print("a: ", a, " b: ", b)
            scores.append(DeepModels(m).cross_eval(X,y,a,b,5))
            aa.append(a)
            bb.append(b)

    min_score = min(scores)
    index = scores.index(min(scores))
    batch = aa[index]
    epochs = bb[index]
    optimized_model = DeepModels(m)
    optimized_model.train(X, y, batch, epochs)
    return optimized_model, min_score, batch, epochs

import torch

X_sj, X_iq, y_sj, y_iq = Preprocess.extract()
sj_pre=[]
# Models
# 0. baseline (2-layer sequential dense layers)
# 1 4-layer sequential
# 2 6-layer sequential
#
m = 7

if m == 7:
    model = DeepModels(m)
    model.train(X_sj, y_sj)
    print("MAE: ", model.cross_eval(X_sj, y_sj))
else:
    model, min_score, batch, epochs = returnOptimizedModel(X_sj, y_sj)
    print(min_score, " ", batch, " ", epochs)

sj_pre = model.prediction(X_sj)
from matplotlib import pyplot as plt
fig1, ax1 = plt.subplots()
# plot sj
plt.plot(range(len(y_sj)), sj_pre)
plt.plot(range(len(y_sj)), y_sj)
plt.show()



#Effect of scaling:
#Effect of differnt things:
#Effect of shuffling:
#Takeaways


