import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from decision_tree import DecisionTree
from datetime import datetime

# === Load and Preprocess Data ===

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/heart.csv") # change path to correct data

data = pd.read_csv(csv_path)
X, y, label_encoder = preprocess_employee_data(data) # replace with new prepocessing

# convert X to a NumPy array here
X = X.values

# # Standardize numeric features (not strictly necessary for decision trees, but doesn't hurt)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# === Train Custom Decision Tree ===
clf = DecisionTree()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

# === Accuracy Function ===
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

acc = accuracy(y_test, predictions)
print("Accuracy:", acc)

# Optional: label info
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
