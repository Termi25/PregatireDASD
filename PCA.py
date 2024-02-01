import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import datasets

iris=datasets.load_iris()
X=iris.data
Y=iris.target

X_std=(X-X.mean())/X.std()
pca=PCA()
pca.fit(X_std)

#Calcul valori componente principale
X_final=pca.transform(X_std)

#Calcul scorul modelului
global_score=pca.score(X_std)

#Calcul scoruri instante
sample_score=pca.score_samples(X_std)

#Export scoruri
scoruri=pd.DataFrame({
    'global_score':global_score,
    'sample_score':sample_score
})
scoruri.to_csv('./OUTPUTS2/ScoruriPCA.csv',sep=',')

#Calcul corelatii factoriale + grafic
corMatrix=pca.components_.T*np.sqrt(pca.explained_variance_)
sns.heatmap(corMatrix,cmap='coolwarm',annot=True)
plt.show()

#Calcul contributii
ctr=X_final**2/np.sum(X_final**2,axis=0)
print('Contributii: ')
print(ctr)

#Calcul cosinusuri
cos_sq=X_final**2/np.sum(X_final**2,axis=1)[:,np.newaxis]
print('Cosinusuri: ')
print(cos_sq)

#Calcul varianta
exp_var=pca.explained_variance_

#Calcul varianta procentuala
exp_var_ratio=pca.explained_variance_ratio_

#Calcul varianta procentuala cumulativa
exp_var_ratio_cum=pca.explained_variance_ratio_.cumsum()

#Grafic
plt.scatter(X_final[:,0],X_final[:,1],c=Y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()