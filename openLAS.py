# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
import dask.dataframe as dd

#%%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 150)
pd.set_option('display.width', 500)


#%%
# Dask 
dfi = dd.read_parquet('../data/interpreted_LAS.pqt')
dfc = dd.read_parquet('../data/cpi_LAS.pqt')
dfr = dd.read_parquet('../data/RTC_LAS.pqt')
dfl = dd.read_parquet('../data/LOGIC_LAS.pqt')

#%%

dfr['DEPTH']=dfr['index']
dfip = dfi.compute()
dfrp = dfr.compute()

coledit = ['TE-021S2', 'TE-030S3', 'TE-031S1','TE-021S1','TE-032S1','TE-028S3','TE-041S1']
colnew = ['TE-021ST2', 'TE-030ST3', 'TE-031ST1','TE-021ST1','TE-032ST1','TE-028ST3','TE-041ST1']
i=0
for i in range(len(coledit)):
    dfip['WELLNAME'] = np.where(dfip.WELLNAME==coledit[i],colnew[i],dfip['WELLNAME'])
    dfrp['WELLNAME'] = np.where(dfrp.WELLNAME==coledit[i],colnew[i],dfrp['WELLNAME'])
dfin = dd.from_pandas(dfip, npartitions=1)
dfrn = dd.from_pandas(dfrp, npartitions=1)

dfm = dd.merge(dfin,dfc, on=['DEPTH','WELLNAME'], how='outer')
dfm = dd.merge(dfm,dfrn, on=['DEPTH','WELLNAME'], how='outer')
dd.to_parquet(dfm,'../data/merged_LAS_v1.pqt',write_metadata_file=False,overwrite=True)



#%%
dflp = dfl.compute()
dflp['WELLNAME']='TE-0'+dflp.WELLNAME.str.split('-').str[1]
coledit = ['TE-021S2', 'TE-030S3', 'TE-031S1','TE-021S1','TE-032S1','TE-028S3','TE-041S1',
        'TE-074S1', 'TE-01', 'TE-02', 'TE-03', 'TE-04', 'TE-05', 'TE-06',
       'TE-07', 'TE-08', 'TE-09', 'TE-026S1', 'TE-028S3_TE', 'TE-030S1', 'TE-036S1',
       'TE-051S2', 'TE-054S1', 'TE-056S1', 'TE-057S1', 'TE-071S1','TE-072_TEMANA_SADDLE', 'TE-073S']
colnew = ['TE-021ST2', 'TE-030ST3', 'TE-031ST1','TE-021ST1','TE-032ST1','TE-028ST3','TE-041ST1',
        'TE-074ST1', 'TE-001', 'TE-002', 'TE-003', 'TE-004', 'TE-005', 'TE-006',
       'TE-007', 'TE-008', 'TE-009', 'TE-026ST1', 'TE-028ST3', 'TE-030ST1', 'TE-036ST1',
       'TE-051ST2', 'TE-054ST1', 'TE-056ST1', 'TE-057ST1', 'TE-071ST1', 'TE-072', 'TE-073ST1']
i=0
for i in range(len(coledit)):
    dflp['WELLNAME'] = np.where(dflp.WELLNAME==coledit[i],colnew[i],dflp['WELLNAME'])
dfln = dd.from_pandas(dflp, npartitions=1)

dfm = dd.merge(dfin,dfln, on=['DEPTH','WELLNAME'], how='outer')
dfm = dd.merge(dfm,dfrn, on=['DEPTH','WELLNAME'], how='outer')
dd.to_parquet(dfm,'../data/merged_LAS.pqt',write_metadata_file=False,overwrite=True)
