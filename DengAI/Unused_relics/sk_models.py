from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


class sk_models:
    def __init__(self, model, cv_k, alpha=0.01, hl=2, maxiter=200):
        if model == 0:
            self.clf = LogisticRegression(random_state=0, solver='saga', multi_class='auto')
        elif model == 1:
            self.clf = DecisionTreeClassifier(random_state=0)
        elif model == 2:
            self.clf = BaggingClassifier(max_samples=0.2, max_features=0.2)
        elif model == 3:
            self.clf = RandomForestClassifier(n_estimators=100)
        elif model == 4:
            self.clf = AdaBoostClassifier(n_estimators=100)
        elif model == 5:
            self.clf = KNeighborsClassifier()
        elif model == 6:
            self.clf = MultinomialNB(alpha=0.40)
        elif model == 7:
            self.clf = VotingClassifier(
                estimators=[('lr', LogisticRegression(random_state=0, solver='saga', multi_class='auto')),
                            ('knb', KNeighborsClassifier()), ('ada', AdaBoostClassifier())],voting='soft')
        elif model == 8:
            self.clf = MLPClassifier(random_state=1, hidden_layer_sizes = (hl,), max_iter = maxiter)
        elif model == 9:
            self.clf = MLPClassifier()
        else:
            self.clf = None
        self.cv_k = cv_k

    def generate_predictions(self, x_train, y_train, x_test):
        """
        Fitting the model
        Fit/train the model on the training set, then predict on the test set
        Returns a list of the predictions made on the test set
        """
        self.clf.fit(x_train, y_train)

        predictions = self.clf.predict(x_test)
        return predictions

    def cross_validate(self, x_train, y_train):
        scores = cross_val_score(self.clf, x_train, y_train, cv=self.cv_k, scoring="neg_mean_absolute_error")
        acc = sum(scores)/len(scores)
        print("Accuracy: ", acc)
        print(scores)

        return acc
