import sys
import pandas as pd
import numpy as np
import random

class KdTree:
    class Node:
        def __init__(self, point, label, left=None, right=None):
            self.point = point
            self.label = label
            self.left = left
            self.right = right

    def __init__(self):
        self.root = None

    def fit(self, X_train, y_train, dimension):
        self.root = self._build_kd_tree(X_train, y_train, dimension)

    def _build_kd_tree(self, points, labels, depth=0, dimension=0):
        if len(points) == 0:
            return None

        k = len(points[0]) if points else 0  # Number of dimensions
        if k == 0:
            return None
        
        axis = dimension % k
        
        if len(points) == 1:  # If only one data point
            return self.Node(points[0], labels[0])
        
        sorted_points = sorted(zip(points, labels), key=lambda x: x[0][axis])  # Sort points based on current axis
        median_index = len(sorted_points) // 2
        median_point, median_label = sorted_points[median_index]
        
        left_points = [point for point, _ in sorted_points[:median_index]]  # Points less than median
        left_labels = [label for _, label in sorted_points[:median_index]]  # Labels corresponding to left points
        right_points = [point for point, _ in sorted_points[median_index + 1:]]  # Points greater than median
        right_labels = [label for _, label in sorted_points[median_index + 1:]]  # Labels corresponding to right points
        
        return self.Node(median_point, median_label,
                         left=self._build_kd_tree(left_points, left_labels, depth + 1, dimension + 1),  # Recursively build left subtree
                         right=self._build_kd_tree(right_points, right_labels, depth + 1, dimension + 1))  # Recursively build right subtree

class KdForest:
    class Node:
        def __init__(self, point, label, left=None, right=None):
            self.point = point
            self.label = label
            self.left = left
            self.right = right

    def __init__(self):
        self.forest = []

    def fit(self, X_train, y_train, dimension_list, rand_seed):
        n_trees = len(dimension_list)
        for j in range(n_trees):
            random.seed(rand_seed+j)
            index_list = [i for i in range(len(X_train))]
            sample_indexes = random.sample(index_list, k=dimension_list[j])
            sampled_data = [(X_train[i], y_train[i]) for i in sample_indexes]
            tree = KdTree()
            tree.fit([data[0] for data in sampled_data], [data[1] for data in sampled_data], dimension_list[j])
            self.forest.append(tree.root)

    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _find_nearest_neighbor(self, tree, query_point, best_dist=float('inf'), best=None):
        if tree is None:
            return best, best_dist
        
        k = len(query_point)
        axis = len(tree.point)
        
        if self._euclidean_distance(query_point, tree.point[:k]) < best_dist:
            best_dist = self._euclidean_distance(query_point, tree.point[:k])
            best = tree
        
        if query_point[axis % k] < tree.point[axis % k]:
            next_branch = tree.left
            opposite_branch = tree.right
        else:
            next_branch = tree.right
            opposite_branch = tree.left
        
        best, best_dist = self._find_nearest_neighbor(next_branch, query_point, best_dist, best)
        
        if abs(tree.point[axis % k] - query_point[axis % k]) < best_dist:
            best, best_dist = self._find_nearest_neighbor(opposite_branch, query_point, best_dist, best)
        
        return best, best_dist

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            labels = []
            for tree in self.forest:
                nearest_neighbor, _ = self._find_nearest_neighbor(tree, sample)
                if nearest_neighbor is not None:
                    labels.append(nearest_neighbor.label)
            if labels:
                predictions.append(max(set(labels), key=labels.count))
            else:
                predictions.append(None)
        return predictions

def read_data(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)
    X = data.iloc[:, :-1].values.astype(float)
    y = data.iloc[:, -1].values
    return X, y

def main(train_file, test_file, random_seed, d_list):
    X_train, y_train = read_data(train_file)
    X_test, y_test = read_data(test_file)

    forest = KdForest()
    forest.fit(X_train, y_train, d_list, random_seed)

    predictions = forest.predict(X_test)
    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python nn_kdforest.py [train] [test] [random_seed] [d_list]")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    random_seed = int(sys.argv[3])
    d_list = list(map(int, sys.argv[4].split(',')))
    main(train_file, test_file, random_seed, d_list)


# References
# [1] Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. Modeling wine preferences by data mining
# from physicochemical properties. Decision support systems 47, 4 (2009), 547–553.