import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from decision_tree import DecisionTree
from datetime import datetime

# === Custom Functions ===

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
    data = data.drop(columns=["Index", "First Name", "Last Name", "Email", "Phone"])

    data["Age"] = data["Date of birth"].apply(calculate_age)
    data = data.drop(columns=["Date of birth"])
    data = data.dropna(subset=["Age"])

    data["Sex"] = data["Sex"].map({"Male": 0, "Female": 1})

    data["JobCategory"] = data["Job Title"].apply(categorize_job_title)
    data = data.drop(columns=["Job Title"])

    y = data["JobCategory"]
    X = data.drop(columns=["JobCategory"])

    X = pd.get_dummies(X, drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le

# === Load and Preprocess Data ===

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/Employee 1000x.csv")

data = pd.read_csv(csv_path)
X, y, label_encoder = preprocess_employee_data(data)

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
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

# === Accuracy Function ===
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

acc = accuracy(y_test, predictions)
print("Accuracy:", acc)

# Optional: label info
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
