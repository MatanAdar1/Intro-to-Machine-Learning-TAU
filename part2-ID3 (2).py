#Matan Adar 322357542
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def entropy(y):
    q = y.sum()/y.size
    return -0.5 *(q *np.log2(q)+(1 - q) * np.log2(1 - q))

def information_gain(X, y, feature_index, threshold):
    left_indices = X[:,feature_index] <= threshold
    right_indices = X[:, feature_index]> threshold
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    p_left = np.sum(left_indices) / len(y)
    p_right = np.sum(right_indices) / len(y)

    return entropy(y)-(p_left*left_entropy + p_right * right_entropy)

def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            gain = information_gain(X, y, feature_index, threshold)
            if gain >best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature: int = feature
        self.threshold: float = threshold
        self.left: Node = left
        self.right: Node = right
        self.value: bool = value

def build_tree(X, y):
    is_trivial = True
    for y_i in y[1:]:
        if y[0] != y_i:
            is_trivial = False
            break
    if is_trivial:
        return Node(value=y[0])

    feature, threshold = best_split(X, y)
    if feature is None:
        most_common = 1 if y.sum() > y.size / 2 else 0
        return Node(value=most_common)

    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold
    left = build_tree(X[left_indices], y[left_indices])
    right = build_tree(X[right_indices], y[right_indices])

    return Node(feature=feature, threshold=threshold, left=left, right=right)


def predict_tree(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)



















def predict(X, tree):
    return np.array([predict_tree(tree, x) for x in X])

def print_tree(node, depth=0):
    indent = "  " * depth
    if node.value is not None:
        print(f"{indent}Leaf: Class {node.value}")
    else:
        print(f"{indent}Feature {node.feature} <= {node.threshold}")
        print(f"{indent}Left:")
        print_tree(node.left, depth + 1)
        print(f"{indent}Right:")
        print_tree(node.right, depth + 1)












if __name__ == "__main__":
    # Loading dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # Spliting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Building the tree using the training set
    tree = build_tree(X_train, y_train)

    # Printing the tree structure
    print("Decision Tree Structure:")
    print_tree(tree)

    # Predicting on the test set
    predictions = predict(X_test, tree)

    # Printing predictions
    print("\nPredictions:", predictions)
    print("Actual:", y_test)

    # Calculating accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

