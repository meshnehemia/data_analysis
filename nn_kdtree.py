import sys
import pandas as pd
import numpy as np

def read_data(file_path):
    """
    Reads data from a file and returns the features and labels.

    Reference:
    2. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. Modeling wine preferences by data mining
       from physicochemical properties. Decision support systems 47, 4 (2009), 547–553.
    """
    try:
        data = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)
        X = data.iloc[:, :-1].values.astype(float)
        y = data.iloc[:, -1].values
        return X, y
    except Exception as e:
        print("Error reading data:", e)
        return None, None

def validate_input(X, y):
    """
    Validates the input data.

    Parameters:
    - X: Feature matrix
    - y: Labels

    Returns:
    - True if the input is valid, False otherwise
    """
    if X is None or y is None:
        print("Error: Input data is missing.")
        return False
    if len(X) != len(y):
        print("Error: Feature matrix and labels have different lengths.")
        return False
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        print("Error: Feature matrix and labels must be NumPy arrays.")
        return False
    if X.ndim != 2:
        print("Error: Feature matrix must be 2-dimensional.")
        return False
    if len(X) == 0:
        print("Error: Feature matrix is empty.")
        return False
    if X.shape[0] != y.shape[0]:
        print("Error: Number of samples in feature matrix and labels do not match.")
        return False
    return True

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def build_kd_tree(points, labels, depth=0, dimension=0):
    if len(points) == 0:
        return None
    
    k = len(points[0])  # Number of dimensions
    axis = dimension % k
    
    if len(points) == 1:  # If only one data point
        return {"point": points[0], "label": labels[0]}
    
    median_index = len(points) // 2
    median_point = points[median_index]
    median_label = labels[median_index]
    
    left_points = points[:median_index]  # Points less than median
    left_labels = labels[:median_index]  # Labels corresponding to left points
    right_points = points[median_index + 1:]  # Points greater than median
    right_labels = labels[median_index + 1:]  # Labels corresponding to right points
    
    return {"point": median_point, "label": median_label,
            "left": build_kd_tree(left_points, left_labels, depth + 1, dimension + 1),  # Recursively build left subtree
            "right": build_kd_tree(right_points, right_labels, depth + 1, dimension + 1)}  # Recursively build right subtree

def find_nearest_neighbor(tree, query_point, best_dist=float('inf'), best=None):
    if tree is None:
        return best, best_dist
    
    k = len(query_point)
    axis = len(tree["point"])
    
    # Calculate the distance between the query point and the current tree node
    dist = euclidean_distance(query_point, tree["point"][:k])
    
    # Update the best neighbor if the current node is closer
    if dist < best_dist:
        best_dist = dist
        best = tree
    
    # Determine which subtree to explore next based on the query point's position
    if query_point[axis % k] < tree["point"][axis % k]:
        next_branch = tree.get("left")
        opposite_branch = tree.get("right")
    else:
        next_branch = tree.get("right")
        opposite_branch = tree.get("left")
    
    # Recursively search the next branch
    best, best_dist = find_nearest_neighbor(next_branch, query_point, best_dist, best)
    
    # Check if we need to search the opposite branch based on the distance to the hyperplane
    if abs(tree["point"][axis % k] - query_point[axis % k]) < best_dist:
        best, best_dist = find_nearest_neighbor(opposite_branch, query_point, best_dist, best)
    
    return best, best_dist

def predict_kd_tree(X_test, tree):
    predictions = []
    for sample in X_test:
        nearest_neighbor, _ = find_nearest_neighbor(tree, sample)
        if nearest_neighbor is not None:
            predictions.append(str(nearest_neighbor["label"]))  # Convert label to string
        else:
            predictions.append(None)
    return predictions

def print_test_results(predictions, test_file_content):
    print("Supplied testing set:")
    print(test_file_content)
    print("\nPredictions:")
    for prediction in predictions:
        print(prediction)

def main(train_file, test_file, dimension):
    # Read training and testing data
    X_train, y_train = read_data(train_file)
    X_test, y_test = read_data(test_file)

    # Validate input data
    if not validate_input(X_train, y_train) or not validate_input(X_test, y_test):
        return

    # Sort training data based on the specified dimension
    sorted_indices = np.argsort(X_train[:, dimension % X_train.shape[1]])
    # Build KD-tree using sorted training data
    kdtree = build_kd_tree(X_train[sorted_indices], y_train[sorted_indices], dimension)

    # Make predictions using the KD-tree
    predictions = predict_kd_tree(X_test, kdtree)

    # Print test results
    with open(test_file, 'r') as file:
        test_file_content = file.read()
    print_test_results(predictions, test_file_content)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python nn_kdtree.py [train] [test] [dimension]")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    dimension = int(sys.argv[3])
    main(train_file, test_file, dimension)