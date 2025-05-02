import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

import knn
def preprocess_heart_data(data):

    # Fill missing values in 'Age' with the median (more robust than the mean)
    data["Age"] = data["Age"].fillna(data["Age"].median())

    # Similarly, fill missing values in 'RestingBP', 'Cholesterol', and 'MaxHR' with the median 
    data["RestingBP"] = data["RestingBP"].fillna(data["RestingBP"].median())
    data["Cholesterol"] = data["Cholesterol"].fillna(data["Cholesterol"].median())
    data["MaxHR"] = data["MaxHR"].fillna(data["MaxHR"].median())
    
    # Convert 'Sex' from string to numeric: male → 0, female → 1
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    # Similarly, convert 'ExerciseAngina', 'ChestPainType', 'RestingECG', and 'ST_Slope' from string to numeric:
    data["ExerciseAngina"] = data["ExerciseAngina"].map({"N": 0, "Y": 1})
    data["ChestPainType"] = data["ChestPainType"].map({"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3})
    data["RestingECG"] = data["RestingECG"].map({"Normal": 0, "ST": 1, "LVH": 2})
    data["ST_Slope"] = data["ST_Slope"].map({"Flat": 0, "Up": 1, "Down": 2})


    y = data["HeartDisease"]
    X = data.drop(columns = "HeartDisease")

    # Label encode the target
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le


# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/heart.csv")

# Load dataset
data = pd.read_csv(csv_path)
X, y, label_encoder = preprocess_heart_data(data)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

for i in range(100):
    # Train KNN model
    classifier = knn.KNN(k=i+1)
    classifier.fit(X_train, y_train)

    # Predict and evaluate
    predictions = classifier.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)

    print("Accuracy for K= " + str(i) + " :", accuracy)


