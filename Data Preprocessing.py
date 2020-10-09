import pandas as pd
import matplotlib.pyplot as plt

# List all the columns of the dataset and report the size of the train part and test part
train_df = pd.read_csv('data/train_Q1.csv')
test_df = pd.read_csv('data/test_Q1.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
train_df.drop(['Loan_ID'], axis=1, inplace=True)
test_df.drop(['Loan_ID'], axis=1, inplace=True)

print(train_df)
print(test_df)

# Find missing value observations and drop them from the data set
print(train_df[train_df.isnull().any(axis=1)])
train_df = train_df.dropna()
train_df.reset_index(drop=True, inplace=True)

# Plot the distribution of all the continuous variables in the data set and check the range of the variable values
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    train_df.reset_index().plot(x='index', y=col, kind='scatter')
plt.show()

# The reasons for doing feature scaling is to normalise the data within a small, specified range, so that it can
# speed up the calculations in machine learning algorithm

# Do feature scaling by MinMax normalization
train_df1 = train_df.apply(lambda x: (x - x.min())/(x.max()-x.min())
if x.name in 'ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History' else x)

# Do feature scaling by feature Standardization
train_df2 = train_df.apply(lambda x: (x - x.mean()) / x.std()
if x.name in 'ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History' else x)

# Choose MinMax normalization in this case
train_df = train_df1

# List all the categorical features and encoding the levels of them into numeric values
categoricalFeatureList = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categoricalFeatureList:
    train_df[col] = train_df[col].astype('category')
    train_df[col] = train_df[col].cat.codes

# Use one-hot-encoding to convert categorical features with n possible values into n binary features,
# with only one active
for col in categoricalFeatureList:
    train_df = pd.concat([train_df, pd.get_dummies(train_df[col], prefix=col)], axis=1)
    train_df.drop([col], axis=1, inplace=True)

print(train_df)