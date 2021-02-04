# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
import missingno as msno
#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score 
from sklearn.metrics import r2_score
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

#%%
df = pd.read_parquet('../data/merged_LAS_v1.pqt/part.0.parquet')
df.tail()

#%%
""" # %%
# Load selected features from SQL into dataframe
dfs = pd.read_sql('SELECT TVDSS,WELLNAME,GR,RTC,NPHIC,RHOBC,SWT,FLUID,KLOG,PHIT_HC FROM all_LAS\
                            WHERE GR IS NOT NULL\
                            AND RTC IS NOT NULL\
                            AND NPHIC IS NOT NULL\
                            AND RHOBC IS NOT NULL\
                            AND TVDSS IS NOT NULL\
                            AND SWT IS NOT NULL\
                            AND KLOG IS NOT NULL\
                            AND PHIT_HC IS NOT NULL\
                            AND FLUID IS NOT NULL', db)#, index_col='index')

 #%%
# Discard columns with more than 90% missing values
cutoff = int(0.1*dfa.shape[0])
dfa.dropna(axis=1,thresh=cutoff,inplace=True)
 """

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
#msno.matrix(dft)

#%%
dfa37 = dfa[dfa.WELLNAME=='TE-037']


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
plt.scatter(dfres.RTC,dfres.RT)
plt.ylim(0,5000)
plt.xlim(0,5000)

#%%
feat = ['WELLNAME','GR_COR','RT','NPHI_COR','RHOB','PHIT','PERM_CH','SWT','RT_RTC']
dfs = dfa[feat]


#%%
# Normalize data and check correlation
dfn = dfs.dropna()
df_norm = (
    dfn.groupby('WELLNAME')
    .apply(SklearnWrapper(preprocessing.MinMaxScaler()))#StandardScaler()))
    .drop("WELLNAME", axis=1)
)
df_norm['WELLNAME'] = dfn['WELLNAME']
print(df_norm.columns)
print(round(df_norm.describe(),2))

print('\nCorrelation for all wells (normalized)')
dfcorr = df_norm.corr(method='kendall')
sn.heatmap(dfcorr, annot=True,fmt='.2f')
plt.show()



# %%
print('Modelling regressors on SWT for Well-45')

""" X = df_norm.iloc[:,0:5]
y = df_norm.iloc[:,5] # Target variable """
X = dfn45.iloc[:,0:5]
y = dfn45.iloc[:,5] # Target variable

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print('Target variable: SWT, Features: TVDSS, GR, RTC, NPHIC, RHOBC') 
# SWT
compareRModels()


#%%
# Tuning SVR parameters for estimator
param = {'kernel':('poly', 'sigmoid','rbf'), 'C':[1, 2, 3]}
GS_CV = GridSearchCV(SVR(),param)
GS_CV.fit(X_train, Y_train)
print(GS_CV.get_params())
#est.predict(X_validation)
print(GS_CV.score(X_validation,Y_validation))
print(GS_CV.best_params_)

#%%
est = SVR(C=3,kernel='poly')
est.fit(X_train, Y_train)
est.score(X_validation,Y_validation)




#%%
def dataQC(df):
# Data QC
    print('Dataframe shape: %d, %d' % df.shape)
    print('\nMissing values, (pct): \n %s' % round(df.isna().sum()/df.shape[0]*100,2).sort_values())
    print('\nDescribe(); \n %s' % round(df.describe(include='all'),2))
    print('\nKurtosis(); \n %s' % round(df.kurt()))
    print('\nSkew(); \n %s' % round(df.skew()))
    df.hist(figsize=(10,10), bins=20)

def compareCModels():#X_train, Y_train):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNNC', KNeighborsClassifier()))    
    models.append(('CART', DecisionTreeClassifier()))    
    models.append(('NB', GaussianNB()))        
    models.append(('SVC', SVC(gamma='auto')))   
    # evaluate each model in turn
    results = []
    names = []
    mean_results = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') #cv=kfold
        results.append(cv_results)
        names.append(name)
        mean_results.append(cv_results.mean())
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    bestModel = models[np.nanargmax(mean_results)][1]
    print('\nBest model: %s' % (bestModel))
    modelScore(bestModel)
    return bestModel

def compareRModels():#X_train, Y_train):
    # Spot Check Algorithms
    models = []    
    models.append(('KNNR', KNeighborsRegressor()))    
    models.append(('DTR', DecisionTreeRegressor()))
    models.append(('ADAB', AdaBoostRegressor()))       
    models.append(('GPR', GaussianProcessRegressor()))    
    #models.append(('SVR', SVR(gamma='auto')))
    models.append(('MLPR', MLPRegressor(random_state=1)))
    # evaluate each model in turn
    results = []
    names = []
    mean_results = []
    for name, model in models:
        cv_results = cross_val_score(model, X_train, Y_train, scoring='r2')
        results.append(cv_results)
        names.append(name)
        mean_results.append(cv_results.mean())
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    bestModel = models[np.nanargmax(mean_results)][1]
    print('\nBest model: %s' % (bestModel))
    modelScore(bestModel)


def modelScore(model):
    #model = GaussianNB()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # Evaluate predictions
    try:
        print('Accuracy score: %f' % (accuracy_score(Y_validation, predictions)))
        #print(confusion_matrix(Y_validation, predictions))
        print('Classification report: \n %s' % (classification_report(Y_validation, predictions)))
    except:
        print('R2: %f' % (model.score(X_train, Y_train)))
        #print(confusion_matrix(Y_validation, predictions))
        #print('Classification report: \n %s' % 

import typing
class SklearnWrapper:
    def __init__(self, transform: typing.Callable):
        self.transform = transform

    def __call__(self, df):
        transformed = self.transform.fit_transform(df.values)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index)