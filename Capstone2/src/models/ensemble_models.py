import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import f1_score
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
# Add path to project source code to access dataset and model code
project_path = '/Users/ishareef7/Springboard/Capstone2'
sys.path.append(project_path)
from src.features import build_dataset

X_train, X_test, y_train, y_test = build_dataset.get_processed_dataset()
X_train2, X_test2, y_train2, y_test2 = build_dataset.get_processed_dataset(crnn_predictions=False)


def baseline_model():
    """Method for creating an Ensemble voting classifier with 3 equally weighted models:
    -Logistic Regression (default parameters)
    -Gradient Boosting (preset parameters
    -SVC (default parameters)"""
    clf1 = LogisticRegression(random_state=7)
    clf2 = GradientBoostingClassifier(n_estimators=750, max_depth=5, max_features='sqrt', random_state=7)
    clf3 = SVC(random_state=7, probability=True)

    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1],
                                   voting='soft', refit=True)
    return eclf


def stacked_model():
    """Method for creating a Stacked Ensemble voting classifier with 3 equally weighted models:
       -Logistic Regression (default parameters)
       -Gradient Boosting (preset parameters
       -SVC (default parameters)"""
    clf1 = LogisticRegression(random_state=7)
    clf2 = GradientBoostingClassifier(n_estimators=750, max_depth=5, max_features='sqrt', random_state=7)
    clf3 = SVC(random_state=7, probability=True)
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1], voting='soft', refit=True)

    return eclf


def train_model(eclf, model):
    assert (model == 'baseline') or (model == 'stacked'), 'Invalid model argument, must be baseline or stacked'

    if model == 'baseline':
        print('Training Baseline Model')
        eclf = eclf.fit(X_train2, y_train2)
    else:
        print('Training Stacked Model')
        eclf = eclf.fit(X_train, y_train)
    return eclf


def get_f1_scores(train_predictions, test_predictions, model, f1_average = 'macro'):
    assert (f1_average == 'macro') or (f1_average is None), 'Invalid f1_average argument. Must be macro or ' \
                                                                    'None'
    assert (model == 'baseline') or (model == 'stacked'), 'Invalid model argument, must be baseline or stacked'

    if model == 'baseline':
        y_train_true = y_train2
        y_test_true = y_test2
    else:
        y_train_true = y_train
        y_test_true = y_test

    if f1_average == 'macro':
        train_f1 = f1_score(y_train_true, train_predictions, average = f1_average)
        test_f1 = f1_score(y_test_true, test_predictions, average=f1_average)

        f1_scores = pd.Series({'Training': train_f1, 'Testing': test_f1})
        f1_scores.name = model
    else:
        labels = ['International', 'Pop', 'Rock', 'Electronic', 'Folk', 'Hip-Hop', 'Experimental', 'Instrumental']
        train_f1 = f1_score(y_train_true, train_predictions, average=f1_average)
        test_f1 = f1_score(y_test_true, test_predictions, average=f1_average)

        f1_dict = {'Training': dict(zip(labels, train_f1)), 'Testing': dict(zip(labels, test_f1))}

        f1_scores = pd.DataFrame(f1_dict)

    return f1_scores


def predict_model(eclf, model):

    if model == 'baseline':
        train_predictions = eclf.predict(X_train2)
        test_predictions = eclf.predict(X_test2)

    else:
        train_predictions = eclf.predict(X_train)
        test_predictions = eclf.predict(X_test)

    return train_predictions, test_predictions



