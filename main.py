from models.MatlabModelImport import MatlabModelImport
import os

def main():
    parameters_directory = r'model_parameters'
    model_filename = r'Model_BEA_refactored.mat'
    model_filepath = os.path.join(parameters_directory, model_filename)
    gc_filename = r'GC_BEA.mat'
    gc_filepath = os.path.join(parameters_directory, gc_filename)
    model_parameters = MatlabModelImport(model_filepath=model_filepath, gc_filepath=gc_filepath)



    return

if __name__ == '__main__':
    main()