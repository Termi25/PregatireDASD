import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
import factor_analyzer as fa_tests

df=pd.read_csv('./Date/bfi.csv')
df.drop(['gender','education','age'],axis=1,inplace=True)
df.dropna(inplace=True)
df_std=(df-df.mean())/df.std()

#Calcul Testul Bartlett de sfericitate
chi_sq,p_value=fa_tests.calculate_bartlett_sphericity(df)

#Calcul Testul KMO
kmo_all,kmo_model=fa_tests.calculate_kmo(df)

#Creare si calculare FactorAnalysis
fa=FactorAnalysis(rotation=None)
fa.fit(df_std)
df_fa=fa.transform(df_std) # df_fa = factorii comuni / variabile latente

#Calcul corelatii factoriale
fa_loadings=fa.components_.T

#Calcul varianta totala
total_var=df_std.var(axis=0).sum()

#Calcul varianta explicata
var_exp=np.sum(fa_loadings**2,axis=0)

#Calcul varianta explicata procentuala
var_exp_proc=var_exp/total_var

#Calcul varianta explicata procentuala cumulativa
var_exp_proc_cum=np.cumsum(var_exp_proc)

#Calcul scoruri fara rotatie
scoruri_nRot=fa.score_samples(df_std)

#Corelograma corelatii factoriale
sns.heatmap(fa_loadings,cmap='coolwarm',annot=True)
plt.show()

#Cercul corelatiilor factoriale
cap=[i for i in range(2436)]
plt.scatter(df_fa[:,0],df_fa[:,1],c=cap)
plt.show()

#Diagrama scoruri
plt.plot(scoruri_nRot,c='m')
plt.show()