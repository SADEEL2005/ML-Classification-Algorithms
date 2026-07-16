# Part 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Part 2: Importing & separating data
dataset = pd.read_csv('ai4i2020.csv')

features_to_drop = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = dataset.drop(columns=features_to_drop)
y = dataset['Machine failure'].values

print("Missing values per column:\n", dataset.isnull().sum())

# Part 3: Encoding Categorical Data 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

# Part 4: Splitting the dataset to Training & Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# stratify=y

# Part 5: Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 5 (Modified): Model building SVM with class_weight='balanced'
classifier_linear = SVC(kernel='linear', random_state=0, class_weight='balanced')
classifier_rbf = SVC(kernel='rbf', random_state=0, class_weight='balanced')
classifier_linear.fit(X_train, y_train)
classifier_rbf.fit(X_train, y_train)

# Part 6: Predicting a new value
# Type, Air temp, Process temp, Rotational speed, Torque, Tool wear
new_data = [['M', 300.0, 310.0, 1500, 40.0, 20]]
new_data_encoded = ct.transform(new_data)
new_data_scaled = sc.transform(new_data_encoded)

print('--- Prediction for New Data ---')
print("Linear SVM prediction:", classifier_linear.predict(new_data_scaled))
print("RBF SVM prediction:", classifier_rbf.predict(new_data_scaled))
print('--'*50)

# Part 7: Predicting the Test set result
y_pred_linear = classifier_linear.predict(X_test)
y_pred_rbf = classifier_rbf.predict(X_test)

# Part 8: Evaluation using Classification Report
print('Linear SVM Classification Report:')
print(classification_report(y_test, y_pred_linear))
print('--'*50)
print('RBF SVM Classification Report:')
print(classification_report(y_test, y_pred_rbf))