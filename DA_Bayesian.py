import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix ,cohen_kappa_score
from sklearn.model_selection import train_test_split

tabel=pd.read_csv('./Date/hernia.csv',index_col=0)
for v in tabel.columns:
    if any(tabel[v].isna()):
        if pd.api.types.is_numeric_dtype(tabel[v]):
            tabel[v].fillna(tabel[v].mean(),inplace=True)
        else:
            tabel[v].fillna(tabel[v].mode()[0],inplace=True)
variabile=list(tabel)
predictori=variabile[:-1]
tinta=variabile[-1]
x_tr,x_ts,y_tr,y_ts=train_test_split(tabel[predictori],tabel[tinta],test_size=0.4)

#Creare si calcul LDA Bayesian
lda_b=GaussianNB()
lda_b.fit(x_tr,y_tr)

#Testare
predictie_b=lda_b.predict(x_ts)

scor_CK=cohen_kappa_score(y_ts,predictie_b)

c=confusion_matrix(y_ts,predictie_b)

tabel_c=pd.DataFrame(c,lda_b.classes_,lda_b.classes_)
tabel_c['Acuratete']=np.round(np.diag(c)*100/np.sum(c,axis=1),3)
acu_medie=tabel_c['Acuratete'].mean()
acu_global=np.round(np.diag(c)*100/len(y_ts),3)

#Predictie
x_apply=pd.read_csv('./Date/hernia_apply.csv',index_col=0)
predictie_apply_b=lda_b.predict(x_apply[predictori])
tabel_predictie=pd.DataFrame(data={'Predictie LDA Bayesiana':predictie_apply_b},index=x_apply.index)