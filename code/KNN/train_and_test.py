import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from knn import KNN
from datetime import datetime

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/Employee 1000x.csv") # change path to correct data

# Load and preprocess data
data = pd.read_csv(csv_path)
X, y, label_encoder = preprocess_employee_data(data) # replace with new prepocessing

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # test_size=0.2 means that we test the accuracy using 20% of the data. The other 80% is used for training 

# Train KNN model
classifier = KNN(k=15) # play with different k values
classifier.fit(X_train, y_train)

# Predict and evaluate
predictions = classifier.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)

print("Accuracy:", accuracy)

# Optional: Show what the labels mean
# print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Compare to sklearn's knn
sk_knn = KNeighborsClassifier(n_neighbors=15)
sk_knn.fit(X_train, y_train)
sk_preds = sk_knn.predict(X_test)
print("Sklearn Accuracy:", np.mean(sk_preds == y_test))