# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np 
import sqlite3 as sql
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

# %%
db = sql.connect('../data/combinedLAS_v1.db')

# %%
df = pd.read_sql('SELECT * FROM all_LAS', db)
df.head()

# %%
dfc = pd.read_sql('SELECT TVDSS,WELLNAME,GR,LLD,NEUT,DEN,DT,SWT,FLUID,KLOG,PHIT_HC FROM all_LAS                         WHERE GR IS NOT NULL                            AND LLD IS NOT NULL                            AND NEUT IS NOT NULL                            AND DEN IS NOT NULL                            AND TVDSS IS NOT NULL', db)
dfc.head()
#dfc.to_sql('selected_LAS', db, if_exists='replace')

# %%
print('\nData grouped by WELLNAME')
dfd = pd.read_sql('SELECT TVDSS,WELLNAME,GR,LLD,NEUT,DEN,DT,SWT,FLUID,KLOG,PHIT_HC FROM selected_LAS GROUP BY WELLNAME',db)
dfd.head()

# %%
db.close()


# %%
print('Correlation for all wells')
dfcorr = dfc.corr(method='spearman')
sn.heatmap(dfcorr, annot=True)
plt.show()
#%%
print('\nCorrelation for WELL-46')
dfc46 = dfc[dfc['WELLNAME']=='WELL-46']
corr46 = dfc46.corr(method='spearman')
sn.heatmap(corr46,annot=True)
plt.show()


#%%
# QC outliers and value range for the features


#%%
# Normalize data
dfc46float = dfc46.drop('WELLNAME', axis=1)
scaler = preprocessing.StandardScaler()
dfn46 = pd.DataFrame(scaler.fit_transform(dfc46float))
dfn46 = dfn46.rename(columns= {0:'TVDSS', 1:'GR', 2: 'LLD', 3: 'NEUT', 4:'DEN', 5:'DT', 6:'SWT', 7:'FLUID', 
                        8: 'KLOG', 9: 'PHIT_HC'})
dfn46.describe()
corr46n = dfn46.corr()
sn.heatmap(corr46n, annot=True)
plt.show()


# %%
dfc46c = dfn46[dfn46.SWT.notna()]
SWT_binned = pd.cut(dfc46c.iloc[:,6],100,retbins=True,labels=range(1,101))
dfc46c['SWT_binned'] = SWT_binned[0]
X = dfc46c.iloc[:,[1,2,4,10]]
y = dfc46c.iloc[:,10] # Target variable

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print('Target variable: SWT, Features: GR, LLD, DEN') 
# SWT
compareModels(X_train, Y_train)



#%%
dfc46c = dfc46[dfc46.SWT.notna()]
SWT_binned = pd.cut(dfc46c.iloc[:,7],10,retbins=True,labels=range(1,11))
dfc46c['SWT_binned'] = SWT_binned[0]
X = dfc46c.iloc[:,[2,3,5,11]]
y = dfc46c.iloc[:,11] # Target variable

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print('Target variable: SWT, Features: GR, LLD, DEN') 
# SWT
compareModels(X_train, Y_train)


# %%
dfc46c = dfc46[dfc46.FLUID.notna()]
X = dfc46c.iloc[:,[2,3,5,8]]
y = dfc46c.iloc[:,8] # Target variable

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print('Target variable: FLUID, Features: GR, LLD, DEN') 
compareModels(X_train, Y_train)



#%%
dfc46c = dfc46[dfc46.PHIT_HC.notna()]
PHIT_binned = pd.cut(dfc46c.iloc[:,10],10,retbins=True,labels=range(1,11))
dfc46c['PHIT_binned'] = PHIT_binned[0]
X = dfc46c.iloc[:,[0,2,3,4,5,11]]
y = dfc46c.iloc[:,11] # Target variable

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print('\nTarget variable: PHIT, Features: TVDSS, GR, LLD, NEUT, DEN') 
compareModels(X_train, Y_train)



#%%
dfc46c = dfc46[dfc46.KLOG.notna()]
KLOG_binned = pd.cut(dfc46c.iloc[:,9],1000,retbins=True,labels=range(1,1001))
dfc46c['KLOG_binned'] = KLOG_binned[0]
X = dfc46c.iloc[:,[0,2,3,4,5,11]]
y = dfc46c.iloc[:,11] # Target variable

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print('\nTarget variable: KLOG, Features: TVDSS, GR, LLD, NEUT, DEN') 
compareModels(X_train, Y_train)


# %%
model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



#%%
def compareModels(X_train, Y_train):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    #models.append(('MLPR', MLPRegressor(random_state=1)))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') #cv=kfold
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

