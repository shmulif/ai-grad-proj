# Grad Project - Supervised Learning  
### Sam Feld, Jay Salgado, Aban Khan, Matt Glennon

This project explores three supervised learning algorithms: **Neural Network**, **Decision Tree**, and **K-Nearest Neighbors (KNN)**. Each algorithm is implemented in its own module under the `code/` directory.

## 📁 Project Structure

```
code/
├── neural_network/
│   └── neural_network.py
├── decision_tree/
│   └── decision_tree.py
├── knn/
│   ├── knn.py
│   └── train_and_test.py
```

---

## 🚀 How to Run the Code

### 1. Neural Network
To run the neural network model:

```bash
python code/neural_network/neural_network.py
```

### 2. Decision Tree
To run the decision tree model:

```bash
python code/decision_tree/decision_tree.py
```

### 3. K-Nearest Neighbors (KNN)
To run the KNN algorithm:

```bash
python code/knn/train_and_test.py
```

This will automatically load the dataset and use the implementation found in `knn.py`.

---

## 📚 Source Attribution

- **Neural Network** and **Decision Tree** models use `scikit-learn`.  
  Documentation: https://scikit-learn.org/
- The **K-Nearest Neighbors** algorithm was implemented from scratch following this video:  
  [StatQuest: KNN](https://www.youtube.com/watch?v=rTEtEy5o3X0)
