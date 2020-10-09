from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from pandas import read_csv

## -- Min Max Normalization
X_train = np.array([[ 1., -1.,  20.],
                    [ 2.5,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = (X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train))
print(X_train)
print(X_scaled)



## -- find nan values
vals2 = np.array([1, np.nan, 3, 4])
print(np.isnan(vals2))


## -- Remove nan values (Elimination Strategy)

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df = df.dropna()
print(df)

## -- Data Splition Process
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train)
print(X_train.shape, X_test.shape)

