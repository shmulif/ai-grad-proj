import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from KNN import KNN
from datetime import datetime

def categorize_job_title(title):
    title = title.lower()

    if any(keyword in title for keyword in ["engineer", "planner", "technician"]):
        return "Engineering"
    elif any(keyword in title for keyword in ["scientist", "physiological", "chemist"]):
        return "Science/Medical"
    elif any(keyword in title for keyword in ["lawyer", "legal"]):
        return "Legal"
    elif any(keyword in title for keyword in ["editor", "multimedia", "writer", "designer"]):
        return "Media/Creative"
    elif any(keyword in title for keyword in ["buyer", "survey", "analyst", "development"]):
        return "Operations"
    elif any(keyword in title for keyword in ["warehouse", "logistics", "supply"]):
        return "Logistics"
    elif any(keyword in title for keyword in ["assistant", "admin", "secretary"]):
        return "Admin/Support"
    else:
        return "Other"

def calculate_age(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%d-%m-%y")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return np.nan

def preprocess_employee_data(data):
    # Drop irrelevant or high-cardinality columns
    data = data.drop(columns=["Index", "First Name", "Last Name", "Email", "Phone"])

    # Convert DOB to Age
    data["Age"] = data["Date of birth"].apply(calculate_age)
    data = data.drop(columns=["Date of birth"])
    data = data.dropna(subset=["Age"])

    # Encode Sex
    data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1})

    # Create new Job Category column
    data["JobCategory"] = data["Job Title"].apply(categorize_job_title)
    data = data.drop(columns=["Job Title"])

    # Extract label and features
    y = data["JobCategory"]
    X = data.drop(columns=["JobCategory"])

    # One-hot encode categorical columns if needed
    X = pd.get_dummies(X, drop_first=True)

    # Label encode the target
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/Employee 1000x.csv")

# Load and preprocess data
data = pd.read_csv(csv_path)
X, y, label_encoder = preprocess_employee_data(data)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train KNN model
classifier = KNN(k=6)
classifier.fit(X_train, y_train)

# Predict and evaluate
predictions = classifier.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)

print("Accuracy:", accuracy)

# Optional: Show what the labels mean
# print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))