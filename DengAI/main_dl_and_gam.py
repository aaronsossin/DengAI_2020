import Preprocess
from DeepModels import DeepModels
from pygam import LinearGAM
import numpy as np
import pandas as pd
import Epidemic
from EpidemicModels import EpidemicModels
from matplotlib import pyplot as plt
import Gridsearch
import torch

# 1. ----------------------- MODEL SELECTION -----------------------

# Models
# 0. 1 Hidden Layer MLP of size 10
# 1. 2 Hidden Layer MLP of sizes 14,7
# 2. 4 Hidden Layer MLP of sizes 20, 15, 10, 5
# 3. 8 Hidden Layer MLP of sizes 70,60,50,...,20,10
# 4. 15 Hidden Layer MLP of sizes 140, 130, ..., 20, 10
# 5. 2 Hidden Layer (1RNN, 1Dense)
# 6. 2 Hidden Layer (1LSTM, 1Dense)
# 7. 1 Hidden RNN Layer
# 8. 1 Hidden LSTM Layer

m = 0  # Placeholder

while True:
    m = int(input(
        "Select model: \n0. 1 Hidden Layer MLP of size 10\n1. 2 Hidden Layer MLP of sizes 14,7\n2. 4 Hidden Layer MLP of sizes 20, 15, 10, 5\n3. 8 Hidden Layer MLP of sizes 70,60,50,...,20,10\n4. 15 Hidden Layer MLP of sizes 140, 130, ..., 20, 10\n5. 2 Hidden Layer (1RNN, 1Dense)\n6. 2 Hidden Layer (1LSTM, 1Dense)\n7. 1 Hidden RNN Layer\n8. 1 Hidden LSTM Layer:\n"))
    if m < 9 and m >= 0:
        break
    else:
        print("incorrect model selection")

answer = input("Combine with binary outbreak classifier? y for yes, _ for no: ")
combine_epidemic = False
if answer == 'y' or answer == 'Y':
    combine_epidemic = True

# 2. --------------------------- PREPROCESSING AND DATA EXTRACTION -----------------

X_sj, X_iq, y_sj, y_iq = Preprocess.extract()
X_sj = Preprocess.scaler(X_sj)
X_iq = Preprocess.scaler(X_iq)

best_score_sj = 0  # Placeholder
best_score_iq = 0

sj_pre = []  # Placeholder
model_sj = DeepModels(0)  # Placeholder
model_iq = DeepModels(0)  # Placeholder

# 3. ---------------------------------- GRIDSEARCH AUTOMATIC PARAMETER OPTIMIZATION ------------
if combine_epidemic:
    # Converting Dengue Cases to True Labels
    y_sje = Epidemic.convertToEpidemic(y_sj, 1)
    y_iqe = Epidemic.convertToEpidemic(y_iq, 1)
    # Outbreak model
    model_sj, max_score, batch, epochs = Gridsearch.returnOptimizedModel_epidemic(1, X_sj, y_sj, 30, 100, 20)
    model_iq, max_score, batch, epochs = Gridsearch.returnOptimizedModel_epidemic(1, X_iq, y_iq, 30, 100, 20)
    # Predicted Outbreak features
    feature_e_sj = model_sj.prediction(X_sj)
    feature_e_iq = model_iq.prediction(X_iq)
    # Adding new feature to featuer space
    X_sj = np.hstack((X_sj, feature_e_sj))
    X_iq = np.hstack((X_iq, feature_e_iq))
    # Testing model with outbreak feature
    print("San Juan Optimizations")
    model_sj, best_score_sj, batch_sj, epochs_sj = Gridsearch.returnOptimizedModel(m, X_sj, y_sj, 21)
    print(best_score_sj, " ", batch_sj, " ", epochs_sj)
    print("Iquitos Optimizations")
    model_iq, best_score_iq, batch_iq, epochs_iq = Gridsearch.returnOptimizedModel(m, X_iq, y_iq, 21)
    print(best_score_iq, " ", batch_iq, " ", epochs_iq)
else:
    print("San Juan Optimizations")
    model_sj, best_score_sj, batch_sj, epochs_sj = Gridsearch.returnOptimizedModel(m, X_sj, y_sj)
    print(best_score_sj, " ", batch_sj, " ", epochs_sj)
    print("Iquitos Optimizations")
    model_iq, best_score_iq, batch_iq, epochs_iq = Gridsearch.returnOptimizedModel(m, X_iq, y_iq)
    print(best_score_iq, " ", batch_iq, " ", epochs_iq)

# 4. --------------------------- FINAL OUTPUT -------------------------------
total_score = (best_score_sj * X_sj.shape[0] + best_score_iq * X_iq.shape[0]) / (X_sj.shape[0] + X_iq.shape[0])
print("STATISTICS: ", "best sj: ", best_score_sj, " best iq: ", best_score_iq, " total: ", total_score)

# PLOT SAN JUAN (With Cross Val)
DeepModels(m, 20).cross_eval_with_plotting("San Juan", X_sj, y_sj, batch_sj, epochs_sj)

# PLOT IQUITOS (With Cross Val)
DeepModels(m, 20).cross_eval_with_plotting("Iquitos", X_iq, y_iq, batch_iq, epochs_iq)

# ------------------------------------ DISCARDED FUNCTIONALITIES --------------------

#Plotting
# sj_pre = np.round(model_sj.prediction(X_sj))
# fig1, ax1 = plt.subplots()
# # plot sj
# plt.plot(range(len(y_sj)), sj_pre, 'm', alpha=0.4)
# plt.plot(range(len(y_sj)), y_sj, 'g')
# plt.show()

#Submission to Kaggle
# y_pred_sj = np.round(model_sj.prediction(sj_test)).astype('int')
# y_pred_iq = np.round(model_iq.prediction(iq_test)).astype('int')
#
# submission = pd.read_csv("submission_format.csv",
#                          index_col=[0, 1, 2])
#
# submission.total_cases = np.concatenate([y_pred_sj, y_pred_iq])
# submission.to_csv("submission.csv")

#
# elif epidemic:
#     y_sj_e = Epidemic.convertToEpidemic(y_sj)
#     y_iq_e = Epidemic.convertToEpidemic(y_iq)
#     fig9, ax9 = plt.subplots()
#     # plot sj
#     plt.plot(range(len(y_sj_e)), y_sj_e * max(y_sj), 'm', alpha=0.4)
#     plt.plot(range(len(y_sj)), y_sj, 'g')
#     plt.xlabel('Week')
#     plt.ylabel('Cases of Dengue')
#     plt.title('Dengue Cases vs. True Outbreaks in San Juan')
#     plt.legend(['True Outbreak Classification', 'Dengue Cases'])
#     plt.show()
#     fig8, ax8 = plt.subplots()
#     # plot sj
#     plt.plot(range(len(y_iq_e)), y_iq_e * max(y_iq), 'm', alpha=0.4)
#     plt.plot(range(len(y_iq)), y_iq, 'g')
#     plt.xlabel('Week')
#     plt.ylabel('Cases of Dengue')
#     plt.title('Dengue Cases vs. True Outbreaks in Iquitos')
#     plt.legend(['True Outbreak Classification', 'Dengue Cases'])
#     plt.show()
#     if m == 7:
#         model_sj = EpidemicModels(m)
#         model_sj.train(X_sj, y_sj)
#         print("MAE: ", model_sj.cross_eval(X_sj, y_sj))
#     else:
#         model_sj, best_score_sj, batch, epochs = Gridsearch.returnOptimizedModel_epidemic(e, X_sj, y_sj_e)
#         print(best_score_sj, " ", batch, " ", epochs)
#         model_iq, best_score_iq, batch, epochs = Gridsearch.returnOptimizedModel_epidemic(e, X_iq, y_iq_e)
#
#         print(best_score_iq, " ", batch, " ", epochs)
#         sj_pr = model_sj.prediction(X_sj)
#         sj_pre = [1 if x >= 0.5 else 0 for x in sj_pr]
#         iq_pr = model_iq.prediction(X_iq)
#         iq_pre = [1 if x >= 0.5 else 0 for x in iq_pr]
#         fig19, ax19 = plt.subplots()
#         # plot sj
#         plt.plot(range(len(y_sj_e)), sj_pre, 'm', alpha=0.5)
#         plt.plot(range(len(y_sj)), y_sj_e, 'g', alpha=0.5)
#         plt.xlabel('Week')
#         plt.ylabel('Outbreak Classification')
#         plt.title('True Outbreak Classes vs. Predicted Outbreak Classes in San Juan')
#         plt.legend(['Predicted', 'True'])
#         plt.show()
#         fig20, ax20 = plt.subplots()
#         # plot sj
#         plt.plot(range(len(y_iq_e)), iq_pre, 'm', alpha=0.5)
#         plt.plot(range(len(y_iq)), y_iq_e, 'g', alpha=0.25)
#         plt.xlabel('Week')
#         plt.ylabel('Outbreak Classification')
#         plt.title('True Outbreak Classes vs. Predicted Outbreak Classes in Iquitos')
#         plt.legend(['Predicted', 'True'])
#         plt.show()
