import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

pd.set_option('display.max_columns', 100, 'display.width', 1000, 'display.max_rows', 1000)
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix



def data_trs(df):
    df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
    df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
    df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))
    df['LastName'] = df.Name.str.split(',').str[0]
    family = df.groupby(df.LastName).Survived
    df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())
    df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)
    df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())
    df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - \
                                        df.Survived.fillna(0), axis=0)
    df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)
    df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)
    df['Alone'] = (df.WomanOrBoyCount == 0)

    # Thanks to https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster
    # "Title" improvement
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')
    # Cabin, Deck
    df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'

    # Thanks to https://www.kaggle.com/erinsweet/simpledetect
    # Fare
    med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    df['Fare'] = df['Fare'].fillna(med_fare)
    # Age
    df['Age'] = df.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))
    # Family_Size
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

    # Thanks to https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis


    df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)
    df.WomanOrBoyCount = df.WomanOrBoyCount.fillna(0)
    df.FamilySurvivedCount = df.FamilySurvivedCount.fillna(0)
    df.Alone = df.Alone.fillna(0)
    encoder = LabelEncoder()

    print('ORIGINAL\n', df.head(), '\n\n', df.isnull().sum())
    df['Sex'] = encoder.fit_transform(df['Sex'])
    df['Embarked'] = df['Embarked'].replace(np.nan, df['Embarked'].value_counts().index[0])
    df['Embarked'] = encoder.fit_transform(df['Embarked'])
    df['Fare'].replace(np.NaN,df['Fare'].mean(),inplace=True)
    df['Embarked'].replace(np.NaN, df['Embarked'].mode(), inplace=True)

    df['Cabin'] = df['Cabin'].replace(np.NaN, 'Other')
    a = df['Cabin'].str.split(' ').tolist()

    df['Cabin_Deck'] = 'Other'

#    print('-------------------------------------------------------------------\n')
    for i in range(0, len(df['Cabin'])):
        if df['Cabin'].loc[i] != 'Other':

            df['Cabin_Deck'].loc[i] = a[i][0][0]
 #   print('-------------------------------------------------------------------\n')

    df['n_recorsd'] = 1

    a = df.groupby('Cabin', as_index=False).sum()
    d = dict(zip(a['Cabin'], a['n_recorsd']))
    df.drop(['n_recorsd'], axis=1, inplace=True)
    df['N_person_per_cabin'] = df['Cabin'].map(d)

    bins_x = [0.0, 30.00, 70.00, 100.00, 600.00]
    binned_names = ['0-30', '30-70', '70-100', '100+']
    df['Fare_cat'] = pd.cut(df['Fare'], bins=bins_x, labels=binned_names, include_lowest=True)
    df['Fare_cat'] = df['Fare_cat'].astype(str)

    tic = df['Ticket'].str.split(' ')
    df['Ticket_type'] = 'None'
    df['Ticket_Number'] = 0
    for i in range(0, len(df['Ticket_type'])):
        if len(tic[i]) > 1:
            df['Ticket_type'].loc[i] = tic.loc[i][0]
            df['Ticket_Number'].loc[i] = tic.loc[i][1]
        else:
            df['Ticket_Number'].loc[i] = tic.loc[i][0]
    df['n_recorsd'] = 1
    a = df.groupby('Ticket', as_index=False).sum()
    d = dict(zip(a['Ticket'], a['n_recorsd']))
    df.drop(['n_recorsd'], axis=1, inplace=True)
    df['N_person_on_ticket'] = df['Ticket'].map(d)

    names = df['Name'].str.split(',', expand=True, n=1)
    df['Surname'] =names[0]
    df['Title'] = names[1].str.split(".", expand=True, n=1)[0]

    df['Missing_Cabin_Info'] = 0
    df['Missing_Age_Info'] = 0
    df['Missing_Cabin_Info'].loc[df['N_person_per_cabin'].isna()] = 1
    df['Missing_Age_Info'].loc[df['Age'].isna()] = 1
    df['N_person_per_cabin'] = df['N_person_per_cabin'].replace(np.nan, 0)
    d = df.groupby(['Title', 'Fare_cat']).mean()

    t=df[df['Age'].isna()]['Title'].tolist()
    f=df[df['Age'].isna()]['Fare_cat'].tolist()
    ids=df[df['Age'].isna()]['PassengerId'].tolist()
    for i in range(0,len(t)):
        df.loc[df[df['PassengerId'] == ids[i]]['Age'].index.item(),'Age']=d.loc[(t[i], f[i])]['Age']

    df['Familysize']=df['SibSp']+df['Parch']
    bins_x_age = [1.0, 14.00, 20.00, 35.00, 50.00, 70.00, 100.00]
    binned_names = ['child', 'teen', 'youg', 'mature', 'over 50', 'old', ]
    df['Age_cat'] = pd.cut(df['Age'], bins=bins_x_age, labels=binned_names, include_lowest=True)
    df['Age_cat'] = df['Age_cat'].astype(str)
    df = df.drop(['PassengerId', 'Cabin', 'Ticket', 'Name','Ticket_Number'], axis=1)
    # for i in df.columns:
    #   if df[i].dtypes == 'object':
    #      print(i)
    #      df[i]=encoder.fit_transform(df[i])

    print('\nAFTER TRASFORM\n', df.head(5), '\n\n')
    return df


def select_best_Tree_features(df, target_var, top_n):
    Y = df[target_var]
    X = df.drop([target_var], axis=1)
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    f = pd.Series(model.feature_importances_, index=X.columns)
    f.nlargest(top_n).plot(kind='barh')
    plt.show()
    print('\nFeatures Scores\n', f.sort_values(ascending=False))
    top_list = f.nlargest(top_n).index.tolist()
    top_list.append(target_var)
    X_fi = df[top_list]
    return X_fi, Y, top_list


def save_file(preds):
    d = pd.read_csv('test.csv')
    y = pd.DataFrame(preds, columns=['Survived'])
    y['PassengerId'] = d['PassengerId']
    columns_titles = ["PassengerId", "Survived"]
    y = y.reindex(columns=columns_titles)
    # a=pd.DataFrame(y,columns=['Survived'],index=X_test['PassengerId'])
    y.to_csv('submission.csv', index=False)

def select_best_2(df,target_col,n):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    chi2_selector = SelectKBest(score_func='', k=n)


    X=df.drop([target_col], axis=1)
    Y=df[target_col]

    chi2_selector.fit(X, Y)

    s=df.iloc[chi2_selector.get_support(indices=True)]
    print(s.head())


   # print(s.head())
    return s


def Polinomial_expansion(dataframes, target_column, degree_pol, trsfm_log=False):
    from sklearn.preprocessing import PolynomialFeatures
    df = dataframes
    target_df = df[target_column]
    df = df.drop(target_column, axis=1)
    p = PolynomialFeatures(degree=degree_pol)
    indexes = df.columns
    df = p.fit_transform(df)

    df = pd.DataFrame(data=df, columns=p.get_feature_names(indexes))
    if trsfm_log == True:
        df = (df + 1).transform(np.log)

    df[target_column] = target_df.copy()

    print('TRAIN\n', df.head(3), '\n')
    print('\n', df.shape, ' ')
    df = df.drop('1', axis=1)
    return df
''''
df = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

df_tot = df.append(X_test, ignore_index=True)

df_tot = data_trs(df_tot)
encoder = LabelEncoder()
for i in df_tot.columns:
   if df_tot[i].dtypes == 'object':
       df_tot[i]=encoder.fit_transform(df_tot[i])
df_t=Polinomial_expansion(df_tot, 'Survived', 2, trsfm_log=False)
df_tot=pd.concat([df_tot,df_t],axis=1)
df_train = df_tot.iloc[:891, :]
df_test = df_tot.iloc[891:, :]
df_test = df_test.drop(['Survived'], axis=1)
select_best_Tree_features(df_train, 'Survived', df_train.shape[1])
print(df_train.head())

'''