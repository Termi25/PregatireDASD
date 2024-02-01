import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_nuts=pd.read_csv('./Date/RO_NUTS.csv')
df_nat=pd.read_csv('./Date/natalitate.csv')

print(df_nat)
print(df_nuts)

natalitati=list(df_nat.columns[2:].values)
corelation =df_nat[natalitati].corr()
corelation.to_csv('eE2_corr.csv',sep=',')

df_merged=df_nuts.merge(df_nat,on='IndicativJudet')
df_merged.to_csv('Merged2.csv',sep=',')


def mascFem(t):
    totalBarbati=t['NascutiMasculinUrban']+t['NascutiMasculinRural']
    totalFemei=t['NascutiFemininUrban']+t['NascutiFemininRural']
    return pd.Series(data=[t['JUDET'],totalBarbati,totalFemei],
                     index=['Judet','Total barbati','Total femei'])


df_totalmascfem=df_merged.apply(func=mascFem,axis=1)
df_totalmascfem.to_csv('eE2_totalMF.csv',sep=',')

print(df_merged[['JUDET','Regiune']])


def gruparePeMedii(t):
    rural=t['NascutiFemininRural']+t['NascutiMasculinRural']
    urban=t['NascutiMasculinUrban']+t['NascutiFemininUrban']
    return pd.Series(data=[rural,urban],
                     index=['Total nascuti rural','Total nascuti urban'])


df_regiuniT_urban_rural=df_merged.groupby(by='Regiune').sum()
df_regiuniT_urban_rural_final=df_regiuniT_urban_rural.apply(func=gruparePeMedii,axis=1)
df_regiuniT_urban_rural_final.to_csv('eE2_regT_urban_rural.csv',sep=',')