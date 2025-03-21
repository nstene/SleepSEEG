import os
import typing as t

from models.matlab_adaptator import MatlabModelImport
from models.sleep_seeg import SleepSEEG

def run_analysis(patient_dir: str, filename: str, password: t.Optional[str] = None, epoch_indices: t.List[int]=None, automatic=False):
    filepath = os.path.join(patient_dir, filename)
    sleep_eeg_instance = SleepSEEG(filepath=filepath, password=password)

    if not automatic:
        sleep_eeg_instance.select_channels()

    # cProfile.run('sleep_eeg_instance.compute_epoch_features()')
    sleep_eeg_instance.extract_epochs_and_compute_features(epoch_indices=epoch_indices, keep_epoch_data=True)
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
    epoch_ext = str(epoch_indices) if epoch_indices else ''
    sleepstage_filename = 'SleepStage_' + filename.split('.')[0] + epoch_ext + '.csv'
    summary_filename = 'Summary_' + filename.split('.')[0] + epoch_ext + '.csv'

    sleep_stage = sleep_eeg_instance.export_sleep_stage_output(output_folder, filename=sleepstage_filename)
    summary = sleep_eeg_instance.export_summary_output(output_folder, filename=summary_filename)

    return sleep_stage, summary

if __name__ == '__main__':
    # EDF FILE
    # filepath='auditory_stimulation_P18_002.edf'

    # MEF3 FILE
    # patient_dir = r'C:\Users\natha\Documents\projects\SleepSEEG\eeg_data\ds003708\ds003708\sub-01\ses-ieeg01\ieeg'
    # filename = 'sub-01_ses-ieeg01_task-ccep_run-01_ieeg.mefd'

    # MED FILE
    patient_dir = r'C:\Users\natha\Documents\projects\SleepSEEG\eeg_data'
    filename = 'var_sf.medd'
    run_analysis(patient_dir=patient_dir, filename=filename, password="L2_password")