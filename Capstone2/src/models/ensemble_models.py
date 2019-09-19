import sys
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Add path to project source code to access dataset and visualization code
mod_path = '/Users/ishareef7/Springboard/Capstone2'
sys.path.append(mod_path)
from src.features import build_dataset
from src.visualization import visualize

# Split datasets for baseline ensemble model
X_train, X_test, y_train, y_test = build_dataset.get_processed_dataset()
# Split datasets for baseline stacked ensemble model
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