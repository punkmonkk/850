import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler



df= pd.read_csv("data/proj1850.csv")
print(df.head())
print(df.info())
print(df.describe())


print(df.isna().any(axis=0).sum()) #how many col have missing value
print(df.isna().any(axis=1).sum()) #how many row have missing value 


df=df.dropna()
df=df.reset_index(drop=True)
num_bins = int(np.ceil(1 + np.log2(len(df))))

#stratified sampling
df["step_data"] = pd.cut(df["Step"],bins=[0., 3.0, 6.0, 9.0, 13.0, np.inf],labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["step_data"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
strat_train_set = strat_train_set.drop(columns=["step_data"], axis = 1)
strat_test_set = strat_test_set.drop(columns=["step_data"], axis = 1)

#df["X_data"] = pd.cut(df["X"],bins=[0., 3.0, 6.0, 9.0, 13.0, np.inf],labels=[1, 2, 3, 4, 5])
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#for train_index, test_index in split.split(df, df["X_data"]):
#    strat_train_set = df.loc[train_index].reset_index(drop=True)
#    strat_test_set = df.loc[test_index].reset_index(drop=True)
#strat_train_set = strat_train_set.drop(columns=["X_data"], axis = 1)
#strat_test_set = strat_test_set.drop(columns=["X_data"], axis = 1)

#df["Y_data"] = pd.cut(df["Y"],bins=[0., 3.0, 6.0, 9.0, 13.0, np.inf],labels=[1, 2, 3, 4, 5])
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#for train_index, test_index in split.split(df, df["Y_data"]):
 #   strat_train_set = df.loc[train_index].reset_index(drop=True)
 #   strat_test_set = df.loc[test_index].reset_index(drop=True)
#strat_train_set = strat_train_set.drop(columns=["Y_data"], axis = 1)
#strat_test_set = strat_test_set.drop(columns=["Y_data"], axis = 1)

#[df["Z_data"] = pd.cut(df["Z"],bins=[0., 3.0, 6.0, 9.0, 13.0, np.inf],labels=[1, 2, 3, 4, 5])
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#for train_index, test_index in split.split(df, df["Z_data"]):
 #   strat_train_set = df.loc[train_index].reset_index(drop=True)
 #   strat_test_set = df.loc[test_index].reset_index(drop=True)
#strat_train_set = strat_train_set.drop(columns=["Z_data"], axis = 1)
#strat_test_set = strat_test_set.drop(columns=["Z_data"], axis = 1)]

#from pandas.plotting import scatter_matrix 
#attributes = ["X","Y","Z","Step"]
#scatter_matrix(df[attributes],figsize=(12,8))

corr_matrix = df.corr()
#plt.matshow(corr_matrix)
sns.heatmap(corr_matrix);

train_y = strat_train_set['Step']
df_X = strat_train_set.drop(columns = ["Step"])

train_X = strat_train_set[['X','Y','Z']]
df_Y = strat_train_set.drop(columns = ["X","Y","Z"])

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(train_X, train_y)

some_data = train_X.iloc[:20]
some_data.columns = train_X.columns
some_step_values = train_y.iloc[:20]


#for i in range(20):
   # some_predictions = model1.predict(some_data.iloc[i].values.reshape(1, -1))
  #  some_actual_values = some_step_values.iloc[i]
   # print("Predictions:", some_predictions)
   # print("Actual values:", some_actual_values)

#model1_prediction = model1.predict(train_X)
from sklearn.metrics import mean_absolute_error
#model1_train_mae = mean_absolute_error(model1_prediction, train_y)
#print("Model 1 training MAE is: ", round(model1_train_mae,2))


from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(n_estimators=30, random_state=42)
model2.fit(train_X, train_y)
model2_predictions = model2.predict(train_X)
model2_train_mae = mean_absolute_error(model2_predictions, train_y)
print("Model 2 training MAE is: ", round(model2_train_mae,2))







