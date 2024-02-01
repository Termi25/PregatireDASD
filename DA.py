import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from sklearn.model_selection import train_test_split
from matplotlib import cm

tabel_inv_test=pd.read_csv('./Date/hernia.csv',index_col=0)
for v in tabel_inv_test.columns:
    if any(tabel_inv_test[v].isna()):
        if pd.api.types.is_numeric_dtype(tabel_inv_test[v]):
            tabel_inv_test[v].fillna(tabel_inv_test[v].mean(),inplace=True)
        else:
            tabel_inv_test[v].fillna(tabel_inv_test[v].mode()[0],inplace=True)
variabile=list(tabel_inv_test)
predictori=variabile[:-1]
tinta=variabile[-1]

#Divizare in set de invatare si de testare
x_tr,x_ts,y_tr,y_ts=train_test_split(tabel_inv_test[predictori],tabel_inv_test[tinta],test_size=0.4)

#Construire si calcul LDA
lda=LinearDiscriminantAnalysis()
lda.fit(x_tr,y_tr)
clase=lda.classes_
q=len(clase)
m=len(predictori)

#Numar functii discriminante
nr_disc=min(q-1,m)

#Scoruri discriminante pentru setul de testare + salvare
z=lda.transform(x_ts)
tz=pd.DataFrame(z,x_ts.index,["Z"+str(i+1) for i in range(nr_disc)])
tz.to_csv('./OUTPUTS2/ScoruriLDA.csv',sep=',')

#Calcul centrii discriminatori
zg=tz.groupby(by=y_ts.values).mean().values

#Testare
predictie_lda=lda.predict(x_ts)

index_CK=cohen_kappa_score(y_ts,predictie_lda)

c=confusion_matrix(y_ts,predictie_lda)

tabel_c=pd.DataFrame(c,clase,clase)
tabel_c['Acuratete']=np.round(np.diag(c)*100/np.sum(c,axis=1),3)
acu_medie=tabel_c['Acuratete'].mean()
acu_global=np.round(np.diag(c)*100/len(y_ts),3)

#Predictie LDA
x_apply=pd.read_csv('./Date/hernia_apply.csv',index_col=0)
predictie_apply=lda.predict(x_apply[predictori])
tabel_predictie_apply=pd.DataFrame(data={"Predictie LDA":predictie_apply}, index=x_apply.index)

#Vizualizare distributie discriminatori
for i in range(nr_disc):
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(1,1,1)
    ax.set_title('Distributie in axa discriminanta '+str(i+1),color='magenta')
    sns.kdeplot(x=z[:,i],hue=y_ts,fill=True,ax=ax)
plt.show()

#Vizualizare instante si centrii dupa axele discriminante
for i in range(nr_disc-1):
    for j in range(i,nr_disc):
        fig=plt.figure(figsize=(8,6))
        ax=fig.add_subplot(1,1,1,aspect=1)
        ax.set_title('Plot instante',color='m')
        ax.set_xlabel("Z"+str(i+1))
        ax.set_ylabel("Z"+str(j+1))
        map=cm.get_cmap("rainbow",q)
        culori=[map(i) for i in np.linspace(0,1,q)]
        for w in range(q):
            x_=z[y_ts==clase[i],i]
            y_=z[y_ts==clase[i],j]
            ax.scatter(x_,y_,color=culori[w],label=clase[i])
        ax.scatter(zg[:,i],zg[:,j],color=culori,alpha=0.5,s=200,marker="s")
        ax.legend()
plt.show()