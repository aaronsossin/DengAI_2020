from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import LSTM
from sk_models import sk_models
from sklearn.model_selection import KFold
import numpy as np
from pygam import LogisticGAM, LinearGAM
from sklearn.metrics import f1_score


class EpidemicModels:

    # Sequential 6 layer neural network
    def returnSequential6(self):
        model = Sequential()
        model.add(Dense(50, input_dim=20, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def returnSequential9(self):
        model = Sequential()
        model.add(Dense(80, input_dim=20, activation='relu'))
        model.add(Dense(70, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def RNN(self):
        model = Sequential()
        model.add(SimpleRNN(2, input_dim=20))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def multi_RNN(self):
        model = Sequential()
        model.add(SimpleRNN(2, input_dim=20))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def baseline(self):
        # Create model
        model = Sequential()
        model.add(Dense(20, input_dim=20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def lstm(self):
        model = Sequential()
        model.add(LSTM(4, input_dim=20))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def multi_lstm(self):
        model = Sequential()
        model.add(LSTM(4, input_dim=20, return_sequences=True))
        model.add(LSTM(4, input_dim=20))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Sequential 4 layer neural network
    def returnSequential4(self):
        model = Sequential()

        model.add(Dense(32, activation='relu', input_dim=20))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def __init__(self, m):
        if m == 0:
            self.model = self.baseline()
            self.type = 0
        elif m == 1:
            self.model = self.returnSequential4()
            self.type = 2
        elif m == 2:
            self.model = self.returnSequential6()
            self.type = 2
        elif m == 3:
            self.model = self.RNN()
            self.type = 1
        elif m == 4:
            self.model = self.multi_RNN()
            self.type = 1
        elif m == 5:
            self.model = self.lstm()
            self.type = 1
        elif m == 6:
            self.model = self.multi_lstm()
            self.type = 1
        elif m == 7:
            self.model = LogisticGAM()
            self.type = 3
        elif m == 8:
            self.model = self.returnSequential9()
            self.type = 2

    def returnModel(self):
        return self.model

    def train(self, X, y, bs=10, epochs=100):
        if self.type == 1:
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        if self.type == 3:
            self.model.gridsearch(X, y)
        else:
            self.model.fit(X, y, batch_size=bs, epochs=epochs, shuffle=True)

    def prediction(self, X):
        if self.type == 1:
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        return self.model.predict(X)

    def cross_eval(self, X, y, bs=10, ep=100, k=5):
        scores = []
        if self.type == 0:
            kf = KFold(n_splits=k, shuffle=True, random_state=0)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.model.fit(X_train, y_train, batch_size=bs, epochs=ep, verbose=0)
                a, score = self.model.evaluate(X_test, y_test, verbose=0)
                scores.append(score)
            return sum(scores) / len(scores)

        elif self.type == 1:
            kf = KFold(n_splits=k, shuffle=False, random_state=0)
            scores = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
                self.model.fit(X_train, y_train, batch_size=bs, epochs=ep, verbose=0)
                score = self.model.evaluate(X_test, y_test, verbose=0)
                scores.append(score)
            return sum(scores) / len(scores)

        elif self.type == 2:
            kf = KFold(n_splits=k, shuffle=True, random_state=0)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.model.fit(X_train, y_train, batch_size=bs, epochs=ep, verbose=0)
                a, score = self.model.evaluate(X_test, y_test, verbose=0)
                print(score)
                scores.append(score)
            return sum(scores) / len(scores)

        elif self.type == 3:
            kf = KFold(n_splits=k, shuffle=False, random_state=0)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.model.gridsearch(X_train, y_train)
                y_pre = self.model.predict(X_test)
                print(y_pre)
                scores.append(f1_score(y_pre, y_test))
            return sum(scores) / len(scores)
