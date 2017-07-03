# importing Needed modules

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Ridge

#reading training dataset of csv file
df_train = pd.read_csv('train.csv') 
df_train.head()
df_train.shape

#taking all the non catg column
nonCat_train_col = ['Item_ID','Category_3','Category_2','Category_1']
nonCat = df_train[nonCat_train_col]
nonCat.head()
nonCat.shape

# taking only categorical column
cat_train_col = ['ID','Datetime']
catTrain = df_train[cat_train_col]
catTrain.shape

# applying Label encoder
le = preprocessing.LabelEncoder()
X_2 = catTrain.apply(le.fit_transform)
X_2.head()
X_2.shape
# appending 'ID' and 'Datetime' with original

X_train = pd.concat([nonCat,X_2], axis=1)
X_train.head()
X_train.shape
X_train["Category_2"].fillna(np.random.uniform(0.0,5.0), inplace = True)
X_train.isnull().sum()
X_train.shape

# extracting label 'y'

y_train_price = df_train['Price']
y_train_price.shape

# training Linear Model
model = Ridge().fit(X_train, y_train_price)

#-------------------------------------------------------------
#taking test data
df_test = pd.read_csv('test.csv')
df_test.head()
df_test.shape

#extarcting feature for test data

nonCat_test_col = ['Item_ID','Category_3','Category_2','Category_1']
nonCatTest = df_test[nonCat_test_col]
nonCatTest.head()

# taking only categorical column test
cat_test_col = ['ID','Datetime']
catTest = df_test[cat_test_col]
catTest.head()

# applying Label encoder for categor data for test data
#le_t = preprocessing.LabelEncoder()
X_2_test = catTest.apply(le.fit_transform)
X_2_test.head()
X_2_test.shape


## appending 'ID' and 'Datetime' with original test 
X_test = pd.concat([nonCatTest,X_2_test], axis=1)
X_test.head()
X_test.shape
X_test["Category_2"].fillna(np.random.uniform(0.0,5.0), inplace = True)
X_test.shape

#predicting values
predictLabel = model.predict(X_test)
predictLabel
