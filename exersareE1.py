import pandas as pd
import numpy as np
import pyarrow as pa
import matplotlib.pyplot as plt


df_industrie=pd.read_csv('./Date/Industrie.csv')
df_localitati=pd.read_csv('./Date/PopulatieLocalitati.csv')
print(df_localitati)
print(df_industrie)

industrii=list(df_industrie.columns[2:].values)
df_merged=df_localitati.merge(df_industrie,how='inner')
# print(df_merged)

df_mergedLocalIndustriiPop=df_merged[['Localitate','Populatie','Judet']+industrii]
print(df_mergedLocalIndustriiPop)


def perCapita(t,vars,pop):
    linie=t[vars].values/t[pop]
    rez=list(linie)
    rez=[t['Localitate']]+rez
    return pd.Series(data=rez,index=['Localitate']+vars)


dataset1=df_mergedLocalIndustriiPop[['Localitate','Populatie']+industrii].apply(func=perCapita,axis=1,vars=industrii,pop='Populatie')
dataset1.to_csv("Output11.csv",sep=',')


def maxCA(t):
    linie=t.values
    maxValue=np.argmax(linie)
    return pd.Series(data=[t.index[maxValue], linie[maxValue]],
                     index=['Activitate dominanta','Cifra de afaceri'])


dataset21=df_mergedLocalIndustriiPop[industrii+['Judet']].groupby(by='Judet').agg(sum)
print(dataset21)
dataset22=dataset21.apply(func=maxCA, axis=1)
print(dataset22)


