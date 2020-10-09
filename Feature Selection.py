import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn import datasets, linear_model

# List all the columns of the dataset and report the size of the train part and test part
A = np.loadtxt(open('data/diabetes.csv', 'r'), delimiter=",", skiprows=1)
A = np.array(A)

x = A[:, 0:8]
y = A[:, 8]
print(x.shape)

# Use Chi squared feature selection approach to select 4 best features of the dataset
x_chi = SelectKBest(chi2, k=4).fit_transform(x, y)
print(x_chi.shape)
print(x_chi)

# Use Recursive Feature Elimination approach to select 4 best features of the dataset
estimator = linear_model.LinearRegression()
rfe = RFE(estimator, n_features_to_select=4, step=1)
rfe.fit(x, y)
x_rec = rfe.transform(x)
print(x_rec.shape)
print(x_rec)
