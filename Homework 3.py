# Homework 3
# DS 480
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import RepeatedKFold


data=pd.read_csv("c1_bdhs.csv")
data=data.dropna()
Target = 'RCA'
Predictors = ['SW', 'MOI', 'YOI', 'DOI_CMC', 'RMOB', 'RYOB', 'RDOB_CMC', 'Region', 'Has_Radio','Has_TV', 'Religion', 'WI', 'MOFB', 'YOB', 'DOB_CMC', 'DOFB_CMC', 'AOR',
               'MTFBI', 'DSOUOM_CMC', 'RW', 'RH', 'RBMI']
X = data[Predictors]
y = data[Target]
X, X, y, y = train_test_split(X, y, test_size=0.4, random_state=1)
lm = LinearRegression()
lm.fit(X, y)
coefficients = dict(zip(["Intercept"] + Predictors, [lm.intercept_] + list(lm.coef_)))
print(coefficients)
lm_pred= lm.predict(X)
print(r2_score(y, lm_pred))

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
ridge_mod = RidgeCV(alphas=np.arange(0.1, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
ridge_mod.fit(X, y)
ridge_coefficients = dict(zip(["Intercept"] + Predictors, [ridge_mod.intercept_] + list(ridge_mod.coef_)))
print(ridge_coefficients)
ridge_pred = ridge_mod.predict(X)
print(r2_score(y, ridge_pred))

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
lasso_model = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1)
lasso_mod = Lasso(alpha=1.0)
lasso_mod.fit(X, y)
lasso_coefficients = dict(zip(["Intercept"] + Predictors, [lasso_mod.intercept_] + list(lasso_mod.coef_)))
print(lasso_coefficients)
lasso_pred = lasso_mod.predict(X)
print(r2_score(y, lasso_pred))

elastic_mod = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_mod.fit(X, y)
elastic_net_coefficients = dict(zip(["Intercept"] + Predictors, [elastic_mod.intercept_] + list(elastic_mod.coef_)))
print(elastic_net_coefficients)
elastic_pred = elastic_mod.predict(X)
print(r2_score(y, elastic_pred))








