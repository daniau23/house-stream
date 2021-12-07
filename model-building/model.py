import streamlit as st
import pandas as pd
import numpy as np
import warnings
from scipy import stats
import pickle

# Making use of column transformer
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.preprocessing import  StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV,train_test_split,ShuffleSplit
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")


df = pd.read_csv("housing-data.zip")

# Data copy
df_copy = df.copy()

df_copy = (
    df_copy
    .assign(
        price=df_copy.price.astype(np.int32),
        area=df_copy.area.astype(np.int16),
        bedrooms=df_copy.bedrooms.astype(np.int8),
        bathrooms=df_copy.bathrooms.astype(np.int8),
        stories=df_copy.stories.astype(np.int8),
        parking=df_copy.parking.astype(np.int8),
        mainroad=df_copy.mainroad.astype('category'),
        guestroom=df_copy.guestroom.astype('category'),
        basement=df_copy.basement.astype('category'),
        hotwaterheating=df_copy.hotwaterheating.astype('category'),
        airconditioning=df_copy.airconditioning.astype('category'),
        prefarea=df_copy.prefarea.astype('category'),
        furnishingstatus=df_copy.furnishingstatus.astype('category'),
    )
)

# Object type
a_object = ['mainroad',
 'guestroom',
 'basement',
 'hotwaterheating',
 'airconditioning',
 'prefarea']


def binary_mapping(x):
    return x.map({'yes': 1, "no": 0})

df_copy[a_object] = df_copy[a_object].apply(binary_mapping)

"""Another way of encoding"""
# from sklearn.preprocessing import LabelEncoder
# encode = LabelEncoder()
# df_copy_1 = df_copy.copy()

# df_copy_1['furnishingstatus'] = encode.fit_transform(df_copy_1['furnishingstatus'])

# print(df_copy)
# print(df_copy_1)




###

# Dropping the first column because only two columns are needed
furnishing_status =  pd.get_dummies(df_copy['furnishingstatus'], drop_first=True)


housing = pd.concat([df_copy,furnishing_status],axis="columns")

housing.drop(columns="furnishingstatus", inplace=True)



# Outlier removal: Quantile-based Flooring and Capping
housing['price'].where(housing['price']<7350000.0,7350000.0,inplace=True)
housing['area'].where(housing['area']<7980.0,7980.0,inplace=True)

# Model building
X = housing.drop(columns=["semi-furnished",'unfurnished','price'])
y = housing['price']


X_train, X_test, y_train,y_test= train_test_split(X,y,random_state=42)

# column_transform = ColumnTransformer(transformers=[('ct',StandardScaler(),[0,1,2,3,9])],remainder='passthrough')
column_transform = ColumnTransformer(transformers=[('ct',StandardScaler(),make_column_selector(dtype_include='number'))],remainder='passthrough')

pipe = Pipeline([('t',column_transform),('clf',LinearRegression())])
ShuffleSplit_cv = ShuffleSplit(test_size=.3,random_state=30)

# parameter grid 
param_grid = {
        't':[MinMaxScaler(),None]
        
}



# # The grid search 
grid = GridSearchCV(pipe,param_grid=param_grid,cv=ShuffleSplit_cv)
grid.fit(X_train,y_train)


# Exporting the algorithm
pickle.dump(grid, open('housing.pkl', 'wb'))