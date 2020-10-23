import pandas as pd

df=pd.read_csv('test.csv')
df_2=pd.read_csv('sub_titanic.csv')
df['Survived']=df_2['Survived']
print(df.head(),'\n\n',df_2.head())
df.to_csv('automl_test.csv',index=False)