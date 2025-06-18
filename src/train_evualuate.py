from preprocess import load_and_preprocess
from sklearn.model_selection import StratifiedKFold,cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

df = load_and_preprocess()

print('Classification')
X = np.array(df.drop(['revenue','Success'],axis=1))
y = np.array(df['Success'])

pipelines = {'Logistic Regression':Pipeline([('scaler',StandardScaler()),
                                             ('clf',LogisticRegression(max_iter=1000))]),
            'Random Forest':Pipeline([('clf',RandomForestClassifier())]),
            'SVM':Pipeline([('scaler',StandardScaler()),
                            ('clf',SVC())]),
            'K-Nearest Neighbors':Pipeline([('scaler',RobustScaler()),
                                            ('clf',KNeighborsClassifier())])}

cv = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)

for name,pipeline in pipelines.items():
    scores = cross_val_score(pipeline,X,y,scoring='accuracy',cv=cv)
    print(f'{name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}')


print('Regression')
X = np.array(df.drop(['revenue','Success'],axis=1))
y = np.array(df['revenue'])
pipelines = {'K-Nearest Neighbors':Pipeline([('scaler',RobustScaler()),
                                            ('clf',KNeighborsRegressor())]),
            'Linear Regression':Pipeline([('scaler',StandardScaler()),
                                          ('clf',LinearRegression())])}

cv = KFold(n_splits=10,shuffle=True,random_state=42)

for name,pipeline in pipelines.items():
    scores = cross_val_score(pipeline,X,y,cv=cv,scoring='r2')
    print(f'{name}: {np.mean(scores)} ± {np.std(scores)}')

'''
Sample Output
Classification
Logistic Regression: 0.7419 ± 0.0136
Random Forest: 0.7623 ± 0.0282
SVM: 0.7100 ± 0.0150
K-Nearest Neighbors: 0.7109 ± 0.0165
Regression
K-Nearest Neighbors: 0.6679114677849681 ± 0.0756965286443442
Linear Regression: 0.6197971772377748 ± 0.06616048796681459
'''
