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

#from pandas.plotting import scatter_matrix 
#attributes = ["X","Y","Z","Step"]
#scatter_matrix(df[attributes],figsize=(12,8))

corr_matrix = df.corr()
#plt.matshow(corr_matrix)
sns.heatmap(corr_matrix);

train_y = strat_train_set['Step']
df_X = strat_train_set.drop(columns = ["Step"])