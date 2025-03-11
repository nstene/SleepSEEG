import scipy
import os.path

import typing as t
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import h5py

class MatlabModelImport:

    def __init__(self, model_filepath, gc_filepath):
        self._model_filepath = model_filepath
        self._gc_filepath = gc_filepath

        gc_mat = scipy.io.loadmat(self._gc_filepath)
        self.GC = gc_mat['GC_BEA']

        model_mat = scipy.io.loadmat(self._model_filepath)
        MODEL_BEA_MATLAB = model_mat['model']

        self._NUMBER_FEATURES = MODEL_BEA_MATLAB.shape[0]
        self._NUMBER_POSSIBILITIES = MODEL_BEA_MATLAB.shape[1]

        model_bea_python = []
        for feature_index in range(0, self._NUMBER_FEATURES):
            model_trees_feature = []
            for possibility_index in range(0, self._NUMBER_POSSIBILITIES):


                children = MODEL_BEA_MATLAB[feature_index, possibility_index]['Children'][0][0].astype(int)
                cut_predictor = [el[0].item() if el[0].size > 0 else None for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['CutPredictor'][0][0]]
                cut_predictor_index = [el[0].astype(int) - 1 for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['CutPredictorIndex'][0][0]]
                cut_point = [el[0].astype(float) for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['CutPoint'][0][0]]
                if possibility_index == 0:
                    node_class = [el[0][0].astype(int) for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['NodeClass'][0][0]]
                else:
                    node_class = [el[0][0].astype(bool) for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['NodeClass'][0][0]]
                class_names = [el[0].astype(int) for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['ClassNames'][0][0]]
                class_probability = MODEL_BEA_MATLAB[feature_index, possibility_index]['ClassProbability'][0][0].astype(float)

                model_tree = ClassificationTree(children=children,
                                                cut_predictor=cut_predictor,
                                                cut_point=cut_point,
                                                node_class=node_class,
                                                class_names=class_names,
                                                class_probability=class_probability,
                                                cut_predictor_index=cut_predictor_index)

                model_trees_feature.append(model_tree)

            model_bea_python.append(model_trees_feature)

        self.models = model_bea_python
        return


class ClassificationTree:

    def __init__(self, children, cut_predictor, cut_point, node_class, class_names, class_probability,
                 cut_predictor_index):
        self.children = children
        self.cut_predictor = cut_predictor
        self.cut_point = cut_point
        self.node_class = node_class
        self.class_names = class_names
        self.class_prob = class_probability
        self.cut_predictor_index = cut_predictor_index

    def custom_predict_proba(self, features):
        """
        Custom predict_proba for MATLAB decision tree.
        Arguments:
        - X: The input features (shape: [n_samples, n_features])
        - split_conditions: A list of split conditions (thresholds, feature indices)
        - leaf_values: A list of leaf node values (class probabilities at each leaf)

        Returns:
        - proba: The predicted class probabilities for each sample (shape: [n_samples, n_classes])
        """
        n_samples = features.shape[0]
        n_classes = len(self.class_names)  # Assuming leaf_values are stored as class probabilities

        proba = np.zeros((n_samples, n_classes))
        node = np.zeros(n_samples, dtype=int)

        # Vectorized traversal: Process all samples simultaneously
        while True:
            # Mask for active (internal) nodes
            is_internal = (node < len(self.cut_predictor_index)) & (self.children[node, 0] > 0)
            if not np.any(is_internal):
                break  # Stop if all samples reached leaf nodes

            # Process only active nodes
            active_nodes = node[is_internal]
            feature_idx = self.cut_predictor_index[active_nodes]  # Feature indices
            threshold = self.cut_point[active_nodes]  # Threshold values
            left_child, right_child = self.children[active_nodes].T  # Left & right child nodes

            # Compute decision (move left/right)
            go_left = features[is_internal, feature_idx] <= threshold
            node[is_internal] = np.where(go_left, left_child, right_child)

        # Assign class probabilities at leaf nodes
        proba[:] = self.class_prob[node]

        return proba

    def predict_proba_deepseek(self, features):
        """
        Predict class probabilities for input X.
        X: Input features (2D array of shape [n_samples, n_features]).
        Returns: Class probabilities (2D array of shape [n_samples, n_classes]).
        """
        n_samples = features.shape[0]
        n_classes = len(self.class_names)
        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            node = 0  # Start at the root node
            while self.children[node, 0] != 0:  # While not a leaf node
                feature_idx = self.cut_predictor_index[node]  # Feature index for splitting
                if features[i, feature_idx] <= self.cut_point[node]:
                    node = self.children[node, 0] - 1  # Go to left child
                else:
                    node = self.children[node, 1] - 1  # Go to right child
            proba[i] = self.class_prob[node]  # Use class probabilities at the leaf node

        return proba

    def build_sklearn_tree(self):
        """
        Rebuild MATLAB tree into a scikit-learn DecisionTreeClassifier.
        """
        N = len(self.cut_point)  # Number of nodes
        X_dummy = np.random.rand(100, len(set(self.cut_predictor)))  # Fake training data
        y_dummy = np.random.randint(0, len(self.class_names), 100)  # Fake labels

        # Train a placeholder tree (will replace its parameters)
        clf = DecisionTreeClassifier(max_depth=434, random_state=True)
        clf.fit(X_dummy, y_dummy)  # Train a dummy tree to set up structure

        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        plot_tree(clf)
        plt.show()

        # Replace tree parameters
        clf.tree_.threshold[:len(self.cut_point)] = self.cut_point[:clf.tree_.threshold.size]
        clf.tree_.feature[:] = self.cut_predictor.astype(int)  # Set feature indices
        clf.tree_.children_left[:] = self.children[:, 0]  # Left child
        clf.tree_.children_right[:] = self.children[:, 1]  # Right child
        clf.tree_.value[:] = self.class_prob.reshape(N, 1, -1)  # Probabilities

        return clf



if __name__ == '__main__':
    parameters_directory = r'C:\Users\natha\Documents\projects\SleepSEEG\model_parameters'
    model_filename = r'Model_BEA_full.mat'
    model_filepath = os.path.join(parameters_directory, model_filename)

    gc_filename = r'GC_BEA.mat'
    gc_filepath = os.path.join(parameters_directory, gc_filename)
    matlab_model_import = MatlabModelImport(model_filepath=model_filepath, gc_filepath=gc_filepath)
    matlab_model_import.models[0][0].custom_predict_proba()
    print('hi')