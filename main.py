import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tree import train_search
from transform import Polinomial_expansion, select_best_2
from transform import data_trs, save_file, select_best_Tree_features
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
from mlxtend.classifier import StackingCVClassifier, EnsembleVoteClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB

pd.set_option('display.max_columns', 100, 'display.width', 1000, 'display.max_rows', 1000)


def one_hot_encode(data, column, drop_first=True):
    dum = pd.get_dummies(data[column], drop_first=drop_first, prefix=column)
    dt = pd.concat([data, dum], axis=1)
    Df = dt.drop([column], axis=1)
    return Df


df = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

df_tot = df.append(X_test, ignore_index=True)

print('DF TOTAL before trs\n', df_tot.head())
df_tot = data_trs(df_tot)
print('DF TOTAL after trs\n', df_tot.head())
df_tot.to_csv('aaa.csv', index=False)

encoder = LabelEncoder()

for i in df_tot.columns:
    if df_tot[i].dtypes == 'object':
        df_tot[i] = encoder.fit_transform(df_tot[i])

# cols_to_one_h_enc = ['Cabin_Deck', 'Fare_cat', 'Pclass', 'Age_cat', 'Ticket_type', 'Title']
# for i in cols_to_one_h_enc:
#    df_tot = one_hot_encode(df_tot, i)

# df_tot=pd.read_csv('complete.csv')
# df_tot=Polinomial_expansion(df_tot, 'Survived', 1, trsfm_log=True)

# df_tot=pd.concat([df_tot,df_t],axis=1)


df_train = df_tot.iloc[:891, :]
df_test = df_tot.iloc[891:, :]

ppp=pd.read_csv('sub_titanic.csv')

y_survived = ppp['Survived']
df_test = df_test.drop(['Survived'], axis=1)
print(df_tot.shape)

df_train, _, __ = select_best_Tree_features(df_train, 'Survived', 24)
# df_train=select_best_2(df_train,'Survived', 50)
print(df_train.head())
df_test = df_test[df_train.drop(['Survived'], axis=1).columns]

print('train df shape ', df_train.shape, '\ntest df shape', df_test.shape)

X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

# aa=train_search(X, y, df_test)

# base learners
# C0=best_classifier
C1 = DecisionTreeClassifier(max_depth=8)
C2 = CatBoostClassifier(verbose=0)
C3 = KNeighborsClassifier()
C4 = BernoulliNB()
C5 = RandomForestClassifier()
C6 = XGBClassifier()
C7 = RidgeClassifier()
C8 = KNeighborsClassifier()
C9 = AdaBoostClassifier()
C10 = MLPClassifier(alpha=1, max_iter=1000)
C11 = RidgeClassifier()
C12 = BaggingClassifier()
C13 = ExtraTreesClassifier()
C14 = XGBRFClassifier()
C15 = GradientBoostingClassifier()
C16 = GaussianNB()
C17 = HistGradientBoostingClassifier()
C18 = KNeighborsClassifier()
C19 = SVC()
C20 = RidgeClassifierCV()
Cm = LogisticRegression(max_iter=3000, C=0.2)
Cm1 = LogisticRegression(max_iter=3000, C=0.4)
Cm2 = LogisticRegression(max_iter=3000, C=0.6)
Cm3 = LogisticRegression(max_iter=3000, C=0.8)
Cm4 = LogisticRegression(max_iter=3000, C=1)
names = ['XGBClassifier', 'RidgeClassifier', 'RidgeClassifierCV', 'HistGradientBoostingClassifier',
         'GradientBoostingClassifier', 'BaggingClassifier', 'ExtraTreesClassifier', 'XGBRFClassifier', 'GaussianNB',
         'AdaBoostClassifier',
         'DecisionTreeClassifier', 'CatBoostClassifier', 'MLPClassifier', 'RandomForestClassifier',
         'KNeighborsClassifier', 'Stack1',
         '', '', '', '', '', '', '']
classifiers = [C1, C2, C3, C4]
classifiers_0 = [C5, C6, C7, C8]
classifiers_1 = [C9, C10, C11, C12]
classifiers_2 = [C13, C14, C15, C16]
classifiers_3 = [C17, C18, C19, C20]

voting1 = EnsembleVoteClassifier(clfs=classifiers)
voting2 = EnsembleVoteClassifier(clfs=classifiers_0)
voting3 = EnsembleVoteClassifier(clfs=classifiers_1)
voting4 = EnsembleVoteClassifier(clfs=classifiers_2)
voting5 = EnsembleVoteClassifier(clfs=classifiers_3)

clfs_t = classifiers + classifiers_0 + classifiers_1 + classifiers_2 + classifiers_3

Stack1 = StackingCVClassifier(classifiers=(classifiers), meta_classifier=Cm)

Stack2 = StackingCVClassifier(classifiers=(classifiers_0), meta_classifier=Cm)

Stack3 = StackingCVClassifier(classifiers=(classifiers_1), meta_classifier=Cm)

Stack4 = StackingCVClassifier(classifiers=(classifiers_2), meta_classifier=Cm)

Stack5 = StackingCVClassifier(classifiers=(classifiers_3), meta_classifier=Cm)

Stack6 = StackingCVClassifier(classifiers=clfs_t, meta_classifier=Cm)

Stack_voting = StackingCVClassifier(classifiers=[voting1, voting2, voting3, voting4, voting5], meta_classifier=Cm)
f = EnsembleVoteClassifier(clfs=[Stack4, Stack6], voting='soft')
result = pd.DataFrame(y_val.values, columns=['target'])
best = [C20, C19, C14, C10, C7, C18,C15, C11]  # C15 C11
worst = [C18, C17, C16, C13, C12, C9]# [C8, C6, C5, C4, C3, C2, C1]
classifiers_g = [C10, C20]

voting_gold = EnsembleVoteClassifier(clfs=classifiers_g, voting='hard')
Stack_gold = StackingCVClassifier(classifiers=classifiers_g, meta_classifier=Cm)
classifiers = [voting_gold, Stack_gold]
# print(result)
n = 0
print('\n')
res = pd.read_csv('sub_titanic.csv')
y_correct = res['Survived'].astype(float)
import random

max_acc = 0
import itertools
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
df_test = scaler.fit_transform(df_test)
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
k = list(itertools.combinations(best, 2))
matrx_res = []
aa = pd.DataFrame(y_val.values, columns=['target'])
aa2 = pd.DataFrame(y_survived.values, columns=['target'])
m = 0
m2 = []
global best_cls
for a in k:
    classifiers_g = list(a)
    voting_gold = EnsembleVoteClassifier(clfs=classifiers_g, voting='hard')
    Stack_gold = StackingCVClassifier(classifiers=classifiers_g, meta_classifier=Cm4)
    classifiers = [voting_gold,Stack_gold]
    for i in classifiers:
        for iteration in range(0, 5):
            i.fit(X_train, y_train)
            y_hat = i.predict(X_val)
            matrx_res.append(y_hat)

            l = i.predict(df_test)
            m2.append(l)
            acc = accuracy_score(y_val, y_hat)

            # print('Accuracy on leaderbord: ',accuracy_score(y_correct,y_hat),'\n')
            if acc > max_acc:
                max_acc = acc
                print(n, 'Accuracy train set: ', acc)
                i.fit(X, y)
                y_hat = i.predict(df_test)
                accu = accuracy_score(y_correct, y_hat)
                combination = classifiers_g

                best_cls = i
                print(i, '\nLeaderBoard Accuracy', accu)

            print(n, 'Accuracy test set: ', acc, ' MAX ACCURACY SO FAR:', max_acc)
        pp = pd.DataFrame(matrx_res).transpose().mean(axis=1)
        aa[m] = pp
        qq = pd.DataFrame(m2).transpose().mean(axis=1)
        aa2[m] = qq
        m += 1
    print(aa2)
    n = n + 1

y_val = aa['target']
aa = aa.drop(['target'], axis=1)
aa2 = aa2.drop(['target'], axis=1)
#print(aa2)
#x_tr, x_test, y_tr, y_test = train_test_split(aa, y_val, test_size=0.1, random_state=43)
classifiers=best+worst+[best_cls]
for i in classifiers:
    print(i)
    for n in range(0,5):
        i.fit(aa, y_val)
        # yp = classifier_fin.predict(x_test)
        y_yup = i.predict(aa2)
        print(n,'accuracy ', accuracy_score(y_survived, y_yup), '\nconfusion\n', confusion_matrix(y_survived, y_yup))



from test import train

# train(10,best_cls,X,y,df_test)
'''''
# print('\n\nStack:', classifiers_g , '\n')
        #scores = cross_val_score(i, X_train, y_train, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        #i.fit(X_train, y_train)
        #s = 'cl' + str(n)
        #y_h = i.predict(X_val)
        # result[s] = y_h
        # print(result)
        # print('Accuracy cross validation: %0.3f' % (scores.mean()))
        # print('Balanced Accuracy test set: %0.3f' % (balanced_accuracy_score(y_val, y_h)))
        # print('Precision test set: %0.3f' % (precision_score(y_val, y_h)))
        # print(confusion_matrix(y_val, y_h))
        # print('Overall accuracy set: %0.3f' % (accuracy_score(y_val, y_h)))
        # print('Recall test set: %0.3f' % (recall_score(y_val, y_h)),)
'''

# print(result.corr())
# Class = Stack_votin
# Class.fit(X, y)
# yy = Class.predict(df_test)

# yy = yy.astype(int)
# save_file(yy)
