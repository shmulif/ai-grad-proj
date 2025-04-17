import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        Store the training data and corresponding labels.
        KNN is a lazy learning algorithm, so no actual training happens here—
        it simply memorizes the data for use during prediction.
        """
        self.X_train = X # the training data
        self.y_train = y # the corresponding labels
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):

        """ compute the distance """

        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]


        """ get closest k """

        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Retrieve labels corresponding to the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        

        """ majority vote """

        # Use a Counter to tally the frequency of each class label among the k nearest neighbors.
        # most_common() returns a sorted list of (label, count) tuples in descending order.
        # We return the label with the highest count (i.e., the most common one).

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]