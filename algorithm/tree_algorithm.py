import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)

        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return DecisionTreeNode(value=unique_classes[np.argmax(class_counts)])

        best_gain = -1
        best_feature_index, best_threshold = None, None
        
        for feature_index in range(n_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            for i in range(1, n_samples):
                if classes[i] != classes[i - 1]:
                    threshold = (thresholds[i] + thresholds[i - 1]) / 2
                    gain = self._information_gain(y, classes[:i], classes[i:])

                    if gain > best_gain:
                        best_gain = gain
                        best_feature_index = feature_index
                        best_threshold = threshold

        left_indices = X[:, best_feature_index] <= best_threshold
        right_indices = X[:, best_feature_index] > best_threshold
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return DecisionTreeNode(best_feature_index, best_threshold, left_child, right_child)

    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        return self._entropy(parent) - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))

    def _entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, X):
        return [self._predict_sample(sample, self.tree) for sample in X]

    def _predict_sample(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    

class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=None, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            if self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            else:
                max_features = n_features

            feature_indices = np.random.choice(n_features, size=max_features, replace=False)
            tree = self._grow_tree(X_sample, y_sample, feature_indices)
            self.trees.append(tree)

    def _grow_tree(self, X, y, feature_indices, depth=0):
        n_samples, n_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)

        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return DecisionTreeNode(value=unique_classes[np.argmax(class_counts)])

        best_gain = -1
        best_feature_index, best_threshold = None, None
        
        for feature_index in feature_indices:
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            for i in range(1, n_samples):
                if classes[i] != classes[i - 1]:
                    threshold = (thresholds[i] + thresholds[i - 1]) / 2
                    gain = self._information_gain(y, classes[:i], classes[i:])

                    if gain > best_gain:
                        best_gain = gain
                        best_feature_index = feature_index
                        best_threshold = threshold

        left_indices = X[:, best_feature_index] <= best_threshold
        right_indices = X[:, best_feature_index] > best_threshold
        left_child = self._grow_tree(X[left_indices], y[left_indices], feature_indices, depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], feature_indices, depth + 1)
        return DecisionTreeNode(best_feature_index, best_threshold, left_child, right_child)

    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        return self._entropy(parent) - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))

    def _entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, X):
        tree_predictions = np.array([self._predict_tree(tree, X) for tree in self.trees])
        return [np.argmax(np.bincount(tree_predictions[:, i])) for i in range(X.shape[0])]

    def _predict_tree(self, node, X):
        predictions = []
        for sample in X:
            predictions.append(self._predict_sample(sample, node))
        return predictions

    def _predict_sample(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)

        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return DecisionTreeNode(value=unique_classes[np.argmax(class_counts)])

        best_gain = -1
        best_feature_index, best_threshold = None, None
        
        for feature_index in range(n_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            for i in range(1, n_samples):
                if classes[i] != classes[i - 1]:
                    threshold = (thresholds[i] + thresholds[i - 1]) / 2
                    gain = self._information_gain(y, classes[:i], classes[i:])

                    if gain > best_gain:
                        best_gain = gain
                        best_feature_index = feature_index
                        best_threshold = threshold

        left_indices = X[:, best_feature_index] <= best_threshold
        right_indices = X[:, best_feature_index] > best_threshold
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return DecisionTreeNode(best_feature_index, best_threshold, left_child, right_child)

    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        return self._entropy(parent) - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))

    def _entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, X):
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        return predictions

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10, max_depth=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y):
        self.estimators_ = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            estimator = self.base_estimator(max_depth=self.max_depth)
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        final_predictions = [np.argmax(np.bincount(predictions[:, i])) for i in range(X.shape[0])]
        return final_predictions

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
