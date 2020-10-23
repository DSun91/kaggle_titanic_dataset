import numpy as np
import pandas as pd
from transform import save_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, jaccard_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from imblearn.ensemble import BalancedRandomForestClassifier

Search_spaces_XGB = {
    "scale_pos_weight": Real(0.0, 2.3),
    'learning_rate': Real(0.8, 1),
    #'min_child_weight': Real(0, 5000),
    'gamma': Real(1, 2),
    'max_depth': Integer(1, 10),
    #'subsample': Real(0.01, 1),
    #'max_delta_step': (0, 200000),
    'colsample_bytree': Real(0.1, 1.0),
    'colsample_bylevel': Real(0.1, 1.0),
    'reg_lambda': Real(1e-6, 10000),
    'reg_alpha': Real(0.000001, 0.1),
    'n_estimators': Integer(300, 400),
    # 'base_score':Real(0.55,0.7),

}


def train(n, X, y):
    bayes_cv_tuner = BayesSearchCV(
        estimator=xgb.XGBClassifier(tree_method='hist',
                                    eval_metric='map',
                                    ),

        search_spaces=Search_spaces_XGB,
        scoring='balanced_accuracy',
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=14),
        n_jobs=-1,
        n_iter=n,
        verbose=0,
        refit=True,
        random_state=42
    )
    Bayes_ = bayes_cv_tuner.fit(X, y)
    return Bayes_


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters
    print('Model #{}\nBest Score: {}\n\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 3)))


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)


# evaluate model
# scores = cross_val_score(classifier, X, y, scoring='balanced_accuracy', cv=cv, n_jobs=-1)


def train_search(X, y, X_testset):
    res = pd.read_csv('sub_titanic.csv')
    y_correct = res['Survived'].astype(float)

    for i in range(2,14):
        Bayes_ = train(i, X, y)
        classifier = Bayes_.best_estimator_
        preds = classifier.predict(X_testset)
        y_v = classifier.predict(X)
        print('accuracy leaderboard', accuracy_score(y_correct.values, preds), '  n:', i)
        print('accuracy training', accuracy_score(y_v, y), '  n:', i,'\n')
        #preds = preds.astype(int)
        #save_file(preds)

    report = pd.DataFrame(Bayes_.cv_results_)

    #print('\nReport:\n', report)
    #print('\nbest estimator:\n', Bayes_.best_params_, '\n')

    # print((y_correct.values),'\n',preds)
    return classifier
    #preds = preds.astype(int)
    #save_file(preds)
