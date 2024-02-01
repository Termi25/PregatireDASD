import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,rgb2hex
from matplotlib import cm
from scipy.cluster.hierarchy import linkage,leaves_list,dendrogram,set_link_color_palette
# from scikitplot.metrics import plot_silhouette
from sklearn.metrics import silhouette_score,silhouette_samples

date=pd.read_csv('ADN/ADN_Tari.csv',index_col=0)
variabile=list(date)[1:]
x=date[variabile].values
h=linkage(x,method='ward')
n=len(date)
clusteri_singleton=leaves_list(h)

#Calcul partitie optima
k=None
m=np.shape(h)[0]
n=m+1
if k is None:
    d=h[1:,2]-h[:m-1,2]
    j=np.argmax(d)
    k=m-j
else:
    j=m-k
threshold=(h[j,2]+h[j+1,2])/2
c=np.arange(n)
for i in range(m-k+1):
    k1=h[i,0]
    k2=h[i,1]
    c[c==k2]=n+i
    c[c==k1]=n+i
coduri=pd.Categorical(c).codes
p=np.array(["C"+str(i+1) for i in coduri]) # <- partitii optime

#Calcul scor Silhouette la nivel de partitie
silh_opt=silhouette_score(x,p)

#Calcul scor Silhouette la nivel de instante
silh_samples=silhouette_samples(x,p)

#Creare rampa de culori in fct de nr de clusteri
cmap=cm.get_cmap('rainbow',k)
culori=[]
for i in range(cmap.N):
    rgba=cmap(i)
    culori.append(rgb2hex(rgba))

#Stabilirea culorilor pentru dendrograma
culori_inst=[]
for i in range(n):
    index_cluster=int(p[clusteri_singleton[i]][1:])-1
    culori_inst.append(culori[index_cluster])

#Ordonare culori dendrograma
ord=np.unique(culori_inst,return_index=True)[1]
culori_dendro=[culori_inst[i] for i in sorted(ord)]

#Plot ierarhie - Dendrograma
fig=plt.figure(figsize=(9,7))
ax=fig.add_subplot(1,1,1)
ax.set_title('Plot ierarhie optima si scor Silhouette: '+str(silh_opt))
if culori is None:
    set_link_color_palette(culori_dendro)
dendrogram(h,ax=ax,color_threshold=threshold,labels=date.index)
plt.savefig('out/dendr_'+str(k))

# #Trasare grafic Silhouette
# fig=plt.figure(figsize=(9,7))
# ax=fig.add_subplot(1,1,1)
# cmap=LinearSegmentedColormap.from_list("cmap",culori,len(culori))
# plot_silhouette(x,p,'Grafic Silhouette',ax=ax,cmap=cmap)
# plt.savefig('Silhouette')

