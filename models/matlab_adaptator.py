import scipy
import os.path

import typing as t
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import h5py

class MatlabModelImport:

    def __init__(self, model_filepath: str, gc_filepath: str):
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


                children = MODEL_BEA_MATLAB[feature_index, possibility_index]['Children'][0][0].astype(int) - 1
                cut_predictor = [el[0].item() if el[0].size > 0 else None
                                 for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['CutPredictor'][0][0]]
                cut_predictor_index = \
                    [el[0].astype(int) - 1
                     for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['CutPredictorIndex'][0][0]]
                cut_point = [el[0].astype(float)
                             for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['CutPoint'][0][0]]
                if possibility_index == 0:
                    node_class = [el[0][0].astype(int)
                                  for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['NodeClass'][0][0]]
                else:
                    node_class = [el[0][0].astype(bool)
                                  for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['NodeClass'][0][0]]
                class_names = [el[0].astype(int)
                               for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['ClassNames'][0][0]]
                class_probability = (
                    MODEL_BEA_MATLAB[feature_index, possibility_index]['ClassProbability'][0][0].astype(float)
                )

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

    def predict_proba(self, features):
        """
        Predict class probabilities for input X.
        X: Input features (2D array of shape [n_samples, n_features]).
        Returns: Class probabilities (2D array of shape [n_samples, n_classes]).
        """
        n_samples = features.shape[0]
        n_classes = len(self.class_names)
        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            node = 0  # root node
            while self.children[node, 0] != -1:  # While not a leaf node
                feature_idx = self.cut_predictor_index[node]  # Feature index for splitting
                if features[i, feature_idx] <= self.cut_point[node]:
                    node = self.children[node, 0]  # Go to left child
                else:
                    node = self.children[node, 1]  # Go to right child
            proba[i] = self.class_prob[node]  # Use class probabilities at the leaf node

        return proba


if __name__ == '__main__':
    pass