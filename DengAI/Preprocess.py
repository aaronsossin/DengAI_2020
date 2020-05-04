from warnings import filterwarnings
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np

def preprocess_data_all_features(data_path, labels_path=None):
        # load data and set index to city, year, weekofyear
        df = pd.read_csv(data_path, index_col=[0, 1, 2])
        print(df.shape)
        # select features we want
        features = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
                    'reanalysis_air_temp_k',
                    'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
                    'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
                    'reanalysis_relative_humidity_percent',
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

def extract():
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

        sj_train, iq_train = preprocess_data_all_features('dengue_features_train.csv',
                                                               labels_path='dengue_labels_train.csv')

        X_sj = sj_train.iloc[:, :-1]
        y_sj = sj_train.iloc[:, -1]
        X_iq = iq_train.iloc[:, :-1]
        y_iq = iq_train.iloc[:, -1]

        return X_sj, X_iq, y_sj, y_iq

def scaler(X):
        return scale(X, axis=0, with_mean=True, with_std=True)

def scalerGAM(X):
        return (X-min(X))/(max(X)-min(X))

def combineWithTest():
    sj_test, iq_test = preprocess_data_all_features('dengue_features_test.csv')
    X_sj, X_iq, y_sj, y_iq = extract()
    X_sj2 = np.vstack((X_sj, sj_test))
    X_iq2 = np.vstack((X_iq, iq_test))
    X_sj2 = scaler(X_sj2)
    X_iq2 = scaler(X_iq2)
    X_sj_after = X_sj2[:X_sj.shape[0]]
    X_iq_after = X_iq2[:X_iq.shape[0]]
    X_sjtest_after = X_sj2[X_sj.shape[0]:]
    X_iqtest_after = X_iq2[X_iq.shape[0]:]
    return X_sj_after, X_iq_after, y_sj, y_iq, X_sjtest_after, X_iqtest_after





