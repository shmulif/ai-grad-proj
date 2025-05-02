import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Loading data set
imported_data = pd.read_csv('code/data/heart.csv')
print(imported_data.head())
print(imported_data.info())
print(imported_data.describe())

# Identifying categories
id_cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
imported_data = pd.get_dummies(imported_data, columns=id_cols, drop_first=True)

# Feature defining
X = imported_data.drop('HeartDisease', axis=1)
y = imported_data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Training!
classifier = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Entropy Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nEntropy Tree Classification Report:\n",
      classification_report(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(classifier, filled=True, feature_names=X.columns,
          class_names=[str(c) for c in classifier.classes_])
plt.title("Decision Tree (Entropy) Visualization")
plt.show()

# Comparing between GINI and Entropy
criteria = ['gini', 'entropy']
train_errors = []
test_errors  = []

for crit in criteria:
    model = DecisionTreeClassifier(
        criterion=crit,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    train_errors.append(1 - train_acc)
    test_errors.append(1 - test_acc)

# Cross-validation

scores = cross_val_score(
    classifier, X, y,
    cv=5,
    scoring='accuracy'
)

print("CV accuracy:", scores.mean(), "±", scores.std())

# Plot error rates
x = range(len(criteria))
width = 0.35

# Visualization
plt.figure(figsize=(6,4))
plt.bar([i - width/2 for i in x], train_errors, width, label='Train Error')
plt.bar([i + width/2 for i in x], test_errors,  width, label='Test Error')
plt.xticks(x, [c.capitalize() for c in criteria])
plt.ylabel('Error Rate')
plt.title('Decision Tree Error: Gini vs. Entropy')
plt.legend()
plt.tight_layout()
plt.show()
