import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bartlett
from sklearn.cross_decomposition import CCA

link2data="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df=pd.read_csv(link2data)
df=df.dropna()

#Normalizare
X=df[['bill_length_mm','bill_depth_mm']]
X_std=(X-X.mean())/X.std()

Y=df[['flipper_length_mm','body_mass_g']]
Y_std=(Y-Y.mean())/Y.std()

#Creare si aplicare analiza canonica
cca=CCA()
cca.fit(X_std,Y_std)
X_c,Y_c=cca.transform(X_std,Y_std) # <- scoruri canonice

#Testul Bartlett
print(bartlett(X_std,Y_std))

#Corelatii canonice
print(np.corrcoef(X_c[:,0],Y_c[:,0]))
plt.scatter(X_c[:,0],Y_c[:,0])
plt.xlabel("Scor canonic 1")
plt.ylabel("Scor canonic 2")
plt.show()

#Corelatii factoriale
corelatii_x=cca.x_loadings_
corelatii_y=cca.y_loadings_

#Biplot
f=plt.figure(figsize=(11,8))
ax=f.add_subplot(1,1,1)
ax.scatter(x=X_c[:,0],y=X_c[:,1],color='Red',label='Set X')
ax.scatter(x=Y_c[:,0],y=Y_c[:,1],color='Blue',label='Set Y')
plt.show()
