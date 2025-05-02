import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(project_root, 'data', 'heart.csv')

df = pd.read_csv(file_path)

y = df['HeartDisease']  # Target

# Convert categorical features to numeric using one-hot encoding
X = pd.get_dummies(df.drop('HeartDisease', axis=1), drop_first=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Train a neural network
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# Plotting the performance of the neural network with different training set sizes
# 10% -> 90% training set size
train_sizes = np.linspace(0.1, 0.9, 9)
train_accuracies = []
test_accuracies = []

for size in train_sizes:
    X_train_part, _, y_train_part, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=300, random_state=42)
    model.fit(X_train_part, y_train_part)

    train_acc = accuracy_score(y_train_part, model.predict(X_train_part))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(train_sizes, test_accuracies, label='Test Accuracy', marker='s')
plt.xlabel('Training Set Size (Proportion)')
plt.ylabel('Accuracy')
plt.title('Neural Network Performance vs. Training Set Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the performance of the neural network with different epochs
epoch_values = [100, 300, 500, 800, 1100, 1500, 2000]
train_accs = []
test_accs = []

for epochs in epoch_values:
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=epochs, random_state=42)
    model.fit(X_train, y_train)

    train_accs.append(accuracy_score(y_train, model.predict(X_train)))
    test_accs.append(accuracy_score(y_test, model.predict(X_test)))

# Find index of highest test accuracy
best_index = np.argmax(test_accs)
best_epoch = epoch_values[best_index]
best_test_acc = test_accs[best_index]

print(f"Best test accuracy: {best_test_acc:.3f} at {best_epoch} epochs")

plt.figure(figsize=(10, 6))
plt.plot(epoch_values, train_accs, label="Train Accuracy", marker='o')
plt.plot(epoch_values, test_accs, label="Test Accuracy", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the performance of the neural network with different hidden layer configurations
layer_configs = [
    (50,),
    (100,),
    (100, 50),
    (128, 64, 32),
    (64, 64)
]

train_accs = []
test_accs = []
labels = []

for config in layer_configs:
    model = MLPClassifier(hidden_layer_sizes=config, activation='relu', max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    train_accs.append(accuracy_score(y_train, model.predict(X_train)))
    test_accs.append(accuracy_score(y_test, model.predict(X_test)))
    labels.append(str(config))

plt.figure(figsize=(10, 6))
x = range(len(labels))
plt.plot(x, train_accs, label="Train Accuracy", marker='o')
plt.plot(x, test_accs, label="Test Accuracy", marker='s')
plt.xticks(x, labels)
plt.xlabel("Hidden Layer Configuration")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Hidden Layers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Cross-validation
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=300, random_state=42)

scores = cross_val_score(model, X_scaled, y, cv=5)

print("Cross-validation scores:", scores)
print(f"Mean accuracy: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")
