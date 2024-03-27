import pandas as pd
import numpy as np
import yaml
import os
import hashlib

with open('./config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

model_file = './Trained_Models/model_all_brain_tumor_data.hdf5'
output_folder = cfg['testing']['output_folder']
model_name = os.path.basename(os.path.normpath(model_file)).split('.hdf5')[0]
out_file = os.path.join(output_folder, 'Predictions_' + model_name + '.csv')

df = pd.read_csv(out_file, sep='\t', header=None, usecols=[0, 1], names=['file_path', 'prediction'])

# split file_paths into patient ID, scan date and scan file name
df['id'] = df['file_path'].str.rsplit('/', 1).str[1].str.split('_', 2).str[0:2].str.join('_')
df['date'] = df['file_path'].str.rsplit('/', 1).str[1].str.split('_', 2).str[2].str.split('__').str[0] # refine this, some file names contain __
df['file_name'] = df['file_path'].str.rsplit('/', 1).str[1].str.split('_', 2).str[2].str.split('__').str[1] # refine this, some file names contain __
df.drop('file_path', axis=1, inplace=True)

# move slice number to own column
df['slice'] = df['file_name'].str.split('_').str[-1].str.split('.').str[0]
df['file_name'] = df['file_name'].str.rsplit('_', 1).str[0] + '.nii.gz'

# returns the first mode if there are multiple modes
def first_mode(x):
    modes = pd.Series.mode(x)
    return modes[0] if len(modes) > 0 else np.nan

# get majority prediction over all slices for each file
df_majority_vote = df.groupby(['id', 'date', 'file_name'])['prediction'].agg(first_mode).reset_index()

# prediction labels from DeepDicomSort github
prediction_labels = {
    1: 'pre-contrast T1-weighted',
    2: 'post-contrast T1-weighted',
    3: 'T2 weighted',
    4: 'Proton density weighted',
    5: 'T2 weighted-FLAIR',
    6: 'Diffusion weighted imaging',
    7: 'Derived imaging',
    8: 'Perfusion weighted-DSC'
}

df_majority_vote['prediction_label'] = df_majority_vote['prediction'].map(prediction_labels)

df_majority_vote.to_csv(os.path.join(output_folder, 'Predictions_' + model_name + '_majority_vote.csv'), index=False)

# compare predictions to ground truth

# hashes to match files from cleaned up dataset to unsorted dataset
def calculate_hash(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()


patient_ids = []
dates = []
filenames = []
filehashes = []

root_dir_clean = '/mnt/share/01_followup_cleanedup'

# Walk through the directory structure
for root, dirs, files in os.walk(root_dir_clean):
    for file in files:
        # fine .nii.gz files and split paths into patient id, date and scan file
        if file.endswith('.nii.gz'):
            # also include file size to match with unsorted dataset
            file_hash = calculate_hash(os.path.join(root, file))
            parts = os.path.normpath(root).split(os.sep)
            patient_id = parts[-2]
            date = parts[-1]

            patient_ids.append(patient_id)
            dates.append(date)
            filenames.append(file)
            filehashes.append(file_hash)

df_cleanedup = pd.DataFrame({
    'patient_id': patient_ids,
    'date': dates,
    'filename': filenames,
    'filehash': filehashes
})

# repeat for unsorted dataset

patient_ids = []
dates = []
filenames = []
filehashes = []

root_dir_unsorted = '/mnt/share/00_patientsdata'

# Walk through the directory structure
for root, dirs, files in os.walk(root_dir_unsorted):
    for file in files:
        # fine .nii.gz files and split paths into patient id, date and scan file
        if file.endswith('.nii.gz'):
            # also include file size to match with unsorted dataset
            file_hash = calculate_hash(os.path.join(root, file))
            parts = os.path.normpath(root).split(os.sep)
            patient_id = parts[-2]
            date = parts[-1]

            patient_ids.append(patient_id)
            dates.append(date)
            filenames.append(file)
            filehashes.append(file_hash)

df_unsorted = pd.DataFrame({
    'patient_id': patient_ids,
    'date': dates,
    'filename': filenames,
    'filehash': filehashes
})

# merge cleaned up and unsorted datasets by file hash to get predictions for cleaned up dataset
df_merged = pd.merge(df_unsorted, df_cleanedup, on=['patient_id', 'date', 'filehash'], how='inner')

# merge again with majority vote predictions
df_merged_predictions = pd.merge(df_merged, df_majority_vote, left_on=['patient_id', 'date', 'filename_x'], right_on=['id', 'date', 'file_name'], how='left')
df_merged_predictions.drop(['id', 'file_name'], axis=1, inplace=True)
df_merged_predictions.rename(columns={'filename_x': 'filename_unsorted', 'filename_y': 'filename_cleanedup'}, inplace=True)

# get ground truth labels from filename
df_merged_predictions['label'] = np.where(df_merged_predictions['filename_cleanedup'].str.contains('t1ce', case=False), 2, 5)
# 15 NaN predictions, probably due to parsing error before when filename contained __ 
df_merged_predictions = df_merged_predictions[np.isfinite(df_merged_predictions['prediction'])]
df_merged_predictions['prediction'] = df_merged_predictions['prediction'].astype(int)

# compute confusion matrix and accuracy
confusion_matrix = pd.crosstab(df_merged_predictions['label'], df_merged_predictions['prediction'], rownames=['Actual'], colnames=['Predicted'])
# confusion matrix for T1ce and T2-FLAIR
confusion_matrix_2_5 = confusion_matrix.drop([1, 3, 6, 7], axis=1)
accuracy = np.diag(confusion_matrix_2_5).sum() / confusion_matrix_2_5.values.sum()
print('Accuracy: ', accuracy)
print(confusion_matrix_2_5)

# save everything to output folder
df_merged_predictions.to_csv(os.path.join(output_folder, 'Predictions_' + model_name + '_merged.csv'), index=False) 
confusion_matrix.to_csv(os.path.join(output_folder, 'Confusion_Matrix_' + model_name + '.csv'))