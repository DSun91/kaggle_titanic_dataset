from skopt import BayesSearchCV
from skopt.utils import Real,Integer,Categorical
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC

params_space = {
     'meta_classifier__C':Real(0.1,1),
    'ridgeclassifier__alpha': Real(0.01,10),
    'ridgeclassifier__class_weight': Categorical(['balanced']),
    'mlpclassifier__activation': Categorical(['relu','logistic','tanh','identity']),
    'mlpclassifier__alpha': Real(0.00001,1),
    #'mlpclassifier__batch_size': Integer(0,300),
    'mlpclassifier__beta_1': Real(0.00001,1),
    'mlpclassifier__beta_2': Real(0.00001,1),
    #'mlpclassifier__epsilon': 1e-08,
    'mlpclassifier__learning_rate': Categorical(['constant','adaptive','invscaling']),
    'mlpclassifier__solver': Categorical(['sgd', 'adam','lbfgs']),
    'mlpclassifier__momentum': Real(0.01,1),
    'mlpclassifier__max_iter': (2000,7000),


}
'''
    
    'mlpclassifier__momentum': 0.9,
    'mlpclassifier__n_iter_no_change': 10,
    'mlpclassifier__nesterovs_momentum': True,
    'mlpclassifier__power_t': 0.5,
    'mlpclassifier__random_state': None,
    'mlpclassifier__shuffle': True,
    'mlpclassifier__solver': 'adam',
    'mlpclassifier__tol': 0.0001,
    'mlpclassifier__validation_fraction': 0.1,
     '''
global searchcv


def train(n, classifier, X, y, df_test):
    res = pd.read_csv('sub_titanic.csv')
    y_correct = res['Survived'].astype(float)
    def on_step(optim_result):

        score = searchcv.best_score_
        print("best score: %s" % score)
        if score >= 0.98:
            print('Interrupting!')
            return True

    print(classifier)
    searchcv = BayesSearchCV(classifier, search_spaces=params_space, n_iter=n, cv=5, refit=True)
    searchcv.fit(X, y, callback=on_step)
    res = pd.read_csv('sub_titanic.csv')
    y_correct = res['Survived'].astype(float)
    best_classifier = searchcv.best_estimator_
    preds = best_classifier.predict(df_test)
    print('accuracy leaderboard', accuracy_score(y_correct.values, preds))


# callback handler
