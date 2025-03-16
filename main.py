from models.matlab_adaptator import MatlabModelImport
import os
from models.sleep_seeg import SleepSEEG

def main():
    filename = 'auditory_stimulation_P18_002.edf'
    data_dir = 'eeg_data'
    filepath = os.path.join(data_dir, filename)
    sleep_eeg_instance = SleepSEEG(filepath=filepath)

    # cProfile.run('sleep_eeg_instance.compute_epoch_features()')
    features = sleep_eeg_instance.compute_epoch_features()
    features_preprocessed, nightly_features = sleep_eeg_instance.preprocess_features()

    parameters_directory = r'model_parameters'
    model_filename = r'Model_BEA_full.mat'
    model_filepath = os.path.join(parameters_directory, model_filename)

    gc_filename = r'GC_BEA.mat'
    gc_filepath = os.path.join(parameters_directory, gc_filename)
    matlab_model_import = MatlabModelImport(model_filepath=model_filepath, gc_filepath=gc_filepath)

    channel_groups = sleep_eeg_instance.cluster_channels(nightly_features=nightly_features,
                                                         gc=matlab_model_import.GC)
    sa, mm = sleep_eeg_instance.score_epochs(features=features_preprocessed,
                                             models=matlab_model_import.models,
                                             channel_groups=channel_groups)

    output_folder = r'results'
    sleepstage_filename = 'SleepStage_' + filename.split('.')[0] + '_refactored.csv'
    summary_filename = 'Summary_' + filename.split('.')[0] + '_refactored.csv'

    sleep_stage = sleep_eeg_instance.export_sleep_stage_output(output_folder, filename=sleepstage_filename)
    summary = sleep_eeg_instance.export_summary_output(output_folder, filename=summary_filename)

    return

if __name__ == '__main__':
    main()