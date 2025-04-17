from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNN import KNN

def preprocess_titanic_data(data):
    """
    Cleans and prepares Titanic dataset for KNN classification.
    - Fills missing values
    - Encodes categorical features ('Sex' and 'Embarked')
    - Drops unnecessary columns

    Parameters:
        data (pd.DataFrame): Raw Titanic DataFrame

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """

    # Drop columns that don't help prediction (text-heavy or IDs)
    data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Fill missing values in 'Age' with the median (more robust than the mean)
    data["Age"] = data["Age"].fillna(data["Age"].median())

    # Fill missing values in 'Embarked' with the most common port (mode)
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    # Convert 'Sex' from string to numeric: male → 0, female → 1
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    # Convert the 'Embarked' column (which contains string values like 'C', 'Q', 'S')
    # into separate binary (0/1) columns using one-hot encoding.
    # For example, a passenger who embarked at 'C' will have 'Embarked_C' = 1 and 'Embarked_S' = 0.
    # We drop one of the three categories ('Q') to avoid redundant information
    # (if C and S are both 0, we know the passenger must have embarked at Q).
    # This transformation allows the KNN model to work with this categorical feature numerically.
    data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)


    return data


# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/Titanic-Dataset.csv")

# Load dataset
data = pd.read_csv(csv_path)

# Preprocess the data
data = preprocess_titanic_data(data)

# Extract features and labels
X = data.drop("Survived", axis=1).values
y = data["Survived"].values

# Scale all feature values so that they are on a similar range.
# # This ensures that features like 'Fare' (which can be in the hundreds) 
# don't overpower smaller-scale features like 'Age' during KNN's distance calculations.
# KNN is sensitive to feature scale, so normalization is essential for fair distance measurement.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sk_knn = KNeighborsClassifier(n_neighbors=15)
sk_knn.fit(X_train, y_train)
sk_preds = sk_knn.predict(X_test)
print("Sklearn Accuracy:", np.mean(sk_preds == y_test))