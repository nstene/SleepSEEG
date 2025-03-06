import scipy
import os.path

import typing as t

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
                cut_point = [el[0].astype(float) for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['CutPoint'][0][0]]
                if possibility_index == 0:
                    node_class = [el[0][0].astype(int) for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['NodeClass'][0][0]]
                else:
                    node_class = [el[0][0].astype(bool) for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['NodeClass'][0][0]]
                class_names = [el[0].astype(int) for el in MODEL_BEA_MATLAB[feature_index, possibility_index]['ClassNames'][0][0]]

                model_tree = ClassificationTree(children=children,
                                                cut_predictor=cut_predictor,
                                                cut_point=cut_point,
                                                node_class=node_class,
                                                class_names=class_names)

                model_trees_feature.append(model_tree)

            model_bea_python.append(model_trees_feature)

        self.models = model_bea_python
        return


class ClassificationTree:

    def __init__(self, children, cut_predictor, cut_point, node_class, class_names):
        self.children = children
        self.cut_predictor = cut_predictor
        self.cut_point = cut_point
        self.node_class = node_class
        self.class_names = class_names


if __name__ == '__main__':
    parameters_directory = r'model_parameters'
    model_filename = r'Model_BEA_refactored.mat'
    model_filepath = os.path.join(parameters_directory, model_filename)

    gc_filename = r'GC_BEA.mat'
    gc_filepath = os.path.join(parameters_directory, gc_filename)
    matlab_model_import = MatlabModelImport(model_filepath=model_filepath, gc_filepath=gc_filepath)
    print('hi')