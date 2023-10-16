import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

#confsuion matrix definition so it can be called when needed
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# 1. Loading the data and display parameters
df = pd.read_csv("data/proj1850.csv")
print(df.head())
print(df.info())
print(df.describe())

sns.pairplot(df, hue='Step', diag_kind='kde')
plt.suptitle('Pairwise Plots of x, y, z colored by step', y=1.02)
plt.show()

# 2. Correlation matrix heatmap
correlation_matrix = df[["X", "Y", "Z", "Step"]].corr().abs()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

# 3. Splitting Data (with stratification)
X = df[["X", "Y", "Z"]]
y = df["Step"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify=y)

# 3.1. Feature Engineering: Creating Polynomial Features for X_train and X_test
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)  # Use the same transformation for X_test

# 3.2. Scaling the data (scale after feature engineering)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# 3.3. Feature Selection based on Importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=12)
rf.fit(X_train_scaled, y_train)

# Get feature importances
importances = rf.feature_importances_

# Create a DataFrame to store feature importances along with their names
feature_names = poly.get_feature_names_out(input_features=["X", "Y", "Z"])
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort features by importance in descending order
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Select the top 3 most important features
top_features = feature_importances['Feature'][:3]

# Get the selected feature indices
selected_feature_indices = [feature_names.tolist().index(feature) for feature in top_features]

# Update X_train_selected and X_test_selected with selected features
X_train_selected = X_train_scaled[:, selected_feature_indices]
X_test_selected = X_test_scaled[:, selected_feature_indices]

# 5. Training and Evaluating Models

# b. Decision Tree
clf_tree = DecisionTreeClassifier(max_depth=10, criterion='entropy')
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_test)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
print("Decision Tree - MAE:", mae_tree)
plot_confusion_matrix(y_test, y_pred_tree, "Decision Tree Confusion Matrix")
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# c. Random Forest
clf_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=12)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("Random Forest - MAE:", mae_rf)
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# d. K-Nearest Neighbors
clf_knn = KNeighborsClassifier(n_neighbors=7)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
print("KNN - MAE:", mae_knn)
plot_confusion_matrix(y_test, y_pred_knn, "KNN Confusion Matrix")
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# e. Support Vector Machine
clf_svm = SVC(C=0.5, kernel='linear', random_state=12)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
print("SVM - MAE:", mae_svm)
plot_confusion_matrix(y_test, y_pred_svm, "SVM Confusion Matrix")
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# 6. Example Prediction
position = [[9.375, 3.0625, 1.51]]
position = np.array(position)
scaled_position = scaler.transform(poly.transform(position))
print("Predicted Step using Decision Tree:", clf_tree.predict(position)[0])
print("Predicted Step using Random Forest:", clf_rf.predict(position)[0])
print("Predicted Step using KNN:", clf_knn.predict(position)[0])
print("Predicted Step using SVM:", clf_svm.predict(position)[0])

sns.boxplot(x='Step', y='X', data=df)
plt.show()

sns.boxplot(x='Step', y='Y', data=df)
plt.show()

sns.boxplot(x='Step', y='Z', data=df)
plt.show()




