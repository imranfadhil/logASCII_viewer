# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

#%%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 150)
pd.set_option('display.width', 500)

def dataQC(df):
# Data QC
    print('Dataframe shape: %d, %d' % df.shape)
    print('\nMissing values, (pct): \n %s' % round(df.isna().sum()/df.shape[0]*100,2).sort_values())
    print('\nDescribe(); \n %s' % round(df.describe(include='all'),2))
    print('\nKurtosis(); \n %s' % round(df.kurt()))
    print('\nSkew(); \n %s' % round(df.skew()))
    df.hist(figsize=(10,10), bins=20)

def compareRModels():#X_train, Y_train):
    # Spot Check Algorithms
    models = []    
    models.append(('KNNR', KNeighborsRegressor()))    
    models.append(('DTR', DecisionTreeRegressor()))
    models.append(('ADAB', AdaBoostRegressor()))       
    models.append(('GPR', GaussianProcessRegressor()))    
    models.append(('SVR', SVR()))
    models.append(('MLPR', MLPRegressor(random_state=1)))
    # evaluate each model in turn
    results = []
    names = []
    mean_results = []
    for name, model in models:
        print('Processing %s '%name)
        cv_results = cross_val_score(model, X_train, Y_train, scoring='r2')
        results.append(cv_results)
        names.append(name)
        mean_results.append(cv_results.mean())
        print('Mean score: %f (std: %f)\n' % (cv_results.mean(), cv_results.std()))
    
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    bestModel = models[np.nanargmax(mean_results)][1]
    print('\nBest model: %s' % (bestModel))


import typing
class SklearnWrapper:
    def __init__(self, transform: typing.Callable):
        self.transform = transform

    def __call__(self, df):
        transformed = self.transform.fit_transform(df)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index) 

#%%
df = pd.read_parquet('../data/merged_LAS_v1.pqt/part.0.parquet')
print(df.shape)
df.tail()

#%%

dfa=df
print('\nMissing values (pct): \n %s' % round(dfa.isna().sum()/dfa.shape[0]*100,2).sort_values())

# List missing data by Well
print('\nNumber  of rows by well: ')
g = dfa.groupby('WELLNAME')
print(g.count().sum(axis=1).sort_values(ascending=False))

#%%
# Display matrix of missing data
arranged =  round(dfa.isna().sum()/dfa.shape[0]*100,2).sort_values()
dft = dfa[arranged.index]
msno.matrix(dft[dft.WELLNAME=='TE-037'])
msno.matrix(dft[dft.WELLNAME=='TE-012'])
msno.matrix(dft[dft.WELLNAME=='TE-072'])
msno.matrix(dft[dft.WELLNAME=='TE-054ST1'])


# %%
# Comparing similar log types 
fig, axs = plt.subplots(2,2, figsize=(10,10))

dfres = dfa[['RTC','RT','RXO']]
print('\n%s' % round(dfres.describe(),2))
sn.heatmap(dfres.corr(),annot=True,fmt='.2f', ax=axs[0][0])

""" dfnphi = dft[['NPHIC','NEUT','NPHI_COR',]]
print('\n%s' %round(dfnphi.describe(),2))
sn.heatmap(dfnphi.corr(),annot=True,fmt='.2f')
"""
dfrhob = dfa[['RHOB','RHOMAA']]
print('\n%s' %round(dfrhob.describe(),2))
sn.heatmap(dfrhob.corr(),annot=True,fmt='.2f',ax=axs[0][1])

dfphit = dfa[['PHIE_HC','PHIT_HC','PHIT','PHIE']]
print('\n%s' %round(dfphit.describe(),2))
sn.heatmap(dfphit.corr(),annot=True,fmt='.2f', ax=axs[1][0])

dfsw = dfa[['SWT','SWE','SWC','SXOE','SXOT']]
print('\n%s' %round(dfsw.describe(),2))
sn.heatmap(dfsw.corr(),annot=True,fmt='.2f', ax=axs[1][1])

plt.show()


# %%
# well with most data
print(dfa.dropna().groupby('WELLNAME').count().sum(axis=1).sort_values())
# Check target variable std
# Does smaller std causes less robustness?
dfa.dropna().groupby('WELLNAME').apply(lambda x: x.PERM_CH.std()).sort_values()


#%%
feat = ['WELLNAME','GR_COR','RT','NPHI_COR','RHOB','PHIT','PERM_CH','SWT','RT_RTC',
'VWATER','VGAS','VOIL','VSHALE','VSAND','VSILT','COAL','FACIES']
dfs = dfa[dfa.WELLNAME=='TE-027'][feat].dropna()
print('Well %s %s'%(dfs.WELLNAME.iloc[0],dfs.shape))
dfs.describe(include='all')

#%%
# Normalize data and check correlation
df_norm = (dfs.groupby(['WELLNAME'])[feat[1:]]
    .apply(SklearnWrapper(preprocessing.MinMaxScaler()))#MinMaxScaler()))#StandardScaler()))
)
df_norm['WELLNAME'] = dfs['WELLNAME']

print('\nCorrelation (normalized data)')
dfcorr = df_norm.corr(method='kendall')
fig = plt.figure(figsize=(15,15))
sn.heatmap(dfcorr, annot=True,fmt='.2f')


# %%
X = df_norm[feat[1:5]]
y = df_norm['SWT']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.40, random_state=1)
X_train2, X_test, Y_train2, Y_test = train_test_split(X_validation, Y_validation, test_size=0.30, random_state=1)

# compareRModels()
# model = KNeighborsRegressor().fit(X_train,Y_train)   
# model = DecisionTreeRegressor().fit(X_train,Y_train)
# model = AdaBoostRegressor().fit(X_train,Y_train)
# model = GaussianProcessRegressor().fit(X_train,Y_train)
# model = SVR().fit(X_train,Y_train)
model = MLPRegressor().fit(X_train,Y_train)
print('Running %s on %s'%(model,y.name))
print('Model score  : %s'%model.score(X_train,Y_train))
print('Test score   : %s'%model.score(X_train2,Y_train2))
y_pred = model.predict(X_test)

fig1, axs = plt.subplots(2,1,figsize=(10,10))
axs[0].set_title('%s Test vs Prediction'%y.name)
axs[0].plot(Y_test.index, Y_test, 'ro')
axs[0].plot(Y_test.index, y_pred, 'g+')
axs[1].scatter(Y_test,y_pred)




#%%
# Tuning MLPR parameters for estimator
param = {'alpha':[.00001, .0001, .001], 'activation': ['identity', 'tanh', 'relu']}
GS_CV = GridSearchCV(MLPRegressor(),param)
GS_CV.fit(X_train, Y_train)
#print(GS_CV.get_params())
#est.predict(X_validation)
print(GS_CV.score(X_validation,Y_validation))
print(GS_CV.best_params_)

#%%
est = SVR(C=3,kernel='poly')
est.fit(X_train, Y_train)
est.score(X_validation,Y_validation)