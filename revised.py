import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression




df= pd.read_csv("data/proj1850.csv")
print(df.head())
print(df.info())
print(df.describe())

# 2. Correlation Matrix
correlation = df.corr().abs()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.show()


# 3. Splitting Data (with stratification)
X = df[["X", "Y", "Z"]]
y = df["Step"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# 4. Training and Evaluating Models

# a. Linear Regression
reg_linear = LinearRegression()
reg_linear.fit(X_train, y_train)
y_pred_linear = reg_linear.predict(X_test)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print("\nLinear Regression - MAE:", mae_linear)


# b. Decision Trees
clf_tree = DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
print("\nDecision Tree - Accuracy:", accuracy_tree)
print("Decision Tree - MAE:", mae_tree)

# c. K-Nearest Neighbors
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
print("\nKNN - Accuracy:", accuracy_knn)
print("KNN - MAE:", mae_knn)

# d. Support Vector Machine
clf_svm = SVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
print("\nSVM - Accuracy:", accuracy_svm)
print("SVM - MAE:", mae_svm)




# 5. Example Prediction
position = [[0, 3.0625, 1.99]]

predicted_step_tree = clf_tree.predict(position)
predicted_step_knn = clf_knn.predict(position)
predicted_step_svm = clf_svm.predict(position)

print("\nPredicted Step using Decision Tree:", predicted_step_tree[0])
print("Predicted Step using KNN:", predicted_step_knn[0])
print("Predicted Step using SVM:", predicted_step_svm[0])

