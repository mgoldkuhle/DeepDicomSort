import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import shutil
import subprocess
from multiprocessing import Pool
from tqdm import tqdm


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def delete_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def convert_DICOM_to_NIFTI(root_dir, dcm2niix_bin):
    # Convert all dicom files in the directory to nifti
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    out_dir = os.path.join(base_dir, 'NIFTI')
    temp_dir = os.path.join(base_dir, 'TEMP')

    create_directory(out_dir)
    delete_directory(temp_dir)

    for root, dirs, files in os.walk(root_dir):
        # if len(files) > 0 and 'MR' in root:
        if len(files) > 0:
            create_directory(temp_dir)
            patient_ID = root.split(root_dir)[1]
            patient_ID = patient_ID.split(os.sep)[1]

            patient_out_folder = os.path.join(out_dir, patient_ID)
            create_directory(patient_out_folder)

            sub_directory_names = root.split(os.path.join(root_dir, patient_ID))[1]
            sub_directory_names = sub_directory_names.split(os.sep)[1:]

            nifti_file_name = '__'.join(sub_directory_names)

            system_command = dcm2niix_bin + ' -6 -z y -a y -b n -d 0 -e n -f ' + nifti_file_name + ' -g n -i n -l y -w 0 -o ' + temp_dir + ' ' + root
            os.system(system_command)

            # dcm2niix doesnt always use the specified file name, this is a work around
            nifti_out_files = glob(os.path.join(temp_dir, '*.nii.gz'))
            if len(nifti_out_files) > 0:
                shutil.move(nifti_out_files[0], os.path.join(patient_out_folder, nifti_file_name + '.nii.gz'))

            delete_directory(temp_dir)

    return out_dir


def move_RGB_images(root_dir, fslval_bin):
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    rgb_dir = os.path.join(base_dir, 'RGB_images')
    create_directory(rgb_dir)

    for root, dirs, files in os.walk(root_dir):
        for i_file in files:
            if '.nii.gz' in i_file:
                image_file = os.path.join(root, i_file)
                system_command = fslval_bin + ' ' + image_file + ' data_type'

                output = subprocess.check_output(system_command, shell=True).decode('utf-8')

                if 'RGB' in output:
                    shutil.move(image_file, os.path.join(rgb_dir, i_file))

    return


def extract_4D_images(root_dir):
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    sort_dir = os.path.join(base_dir, 'DeepDicomSorted')
    data_dir = os.path.join(sort_dir, 'DATA')
    create_directory(data_dir)
    create_directory(sort_dir)
    out_4D_file = os.path.join(sort_dir, 'Scans_4D.txt')

    with open(out_4D_file, 'w') as the_file:
        for root, dirs, files in os.walk(root_dir):
            print('Extracting 4D... Current root: ' + root)

            for i_file in files:
                if '.nii.gz' in i_file:
                    image_file = os.path.join(root, i_file)

                    image = sitk.ReadImage(image_file, sitk.sitkFloat32)
                    if image.GetDimension() == 4:
                        file_name = i_file.split('.nii.gz')[0]
                        image_size = list(image.GetSize())
                        image_size[3] = 0
                        image = sitk.Extract(image,
                                             size=image_size,
                                             index=[0, 0, 0, 0])

                        sitk.WriteImage(image, image_file)

                        the_file.write(file_name + '\n')
    return out_4D_file


def reorient_to_std(root_dir, fslreorient_bin):
    for root, dirs, files in os.walk(root_dir):
        for i_file in files:
            if '.nii.gz' in i_file:
                full_file = os.path.join(root, i_file)

                command = fslreorient_bin + ' ' + full_file + ' ' + full_file
                os.system(command)
    return


def resample_images(root_dir, resample_size):
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    sort_dir = os.path.join(base_dir, 'DeepDicomSorted')
    out_dir = os.path.join(sort_dir, 'NIFTI_RESAMPLED')
    
    create_directory(sort_dir)
    create_directory(out_dir)

    for root, dirs, files in os.walk(root_dir):
        print('Resampling... Current root: ' + root)
        for i_file in files:
            if '.nii.gz' in i_file:
                patient_data = root.split(root_dir)[1]
                patient_ID = patient_data.split(os.sep)[0]
                patient_date = patient_data.split(os.sep)[1]

                patient_out_folder = os.path.join(out_dir, patient_ID)
                patient_date_folder = os.path.join(patient_out_folder, patient_date)

                create_directory(patient_out_folder)
                create_directory(patient_date_folder)

                out_file = os.path.join(patient_date_folder, i_file)

                image = sitk.ReadImage(os.path.join(root, i_file), sitk.sitkFloat32)

                original_size = image.GetSize()
                original_spacing = image.GetSpacing()

                if len(original_size) == 2:
                    original_size = original_size + (1, )
                if len(original_spacing) == 2:
                    original_spacing = original_spacing + (1.0, )

                new_spacing = [original_size[0]*original_spacing[0]/resample_size[0],
                               original_size[1]*original_spacing[1]/resample_size[1],
                               original_size[2]*original_spacing[2]/resample_size[2]]

                ResampleFilter = sitk.ResampleImageFilter()

                ResampleFilter.SetInterpolator(sitk.sitkBSpline)

                ResampleFilter.SetOutputSpacing(new_spacing)
                ResampleFilter.SetSize(resample_size)

                ResampleFilter.SetOutputDirection(image.GetDirection())
                ResampleFilter.SetOutputOrigin(image.GetOrigin())
                ResampleFilter.SetOutputPixelType(image.GetPixelID())
                ResampleFilter.SetTransform(sitk.Transform())

                resampled_image = ResampleFilter.Execute(image)

                sitk.WriteImage(resampled_image, out_file)
    return out_dir


def slice_images(root_dir):
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    out_dir = os.path.join(base_dir, 'NIFTI_SLICES')

    create_directory(out_dir)

    for root, dirs, files in os.walk(root_dir):
        print('Slicing... Current root: ' + root)
        for i_file in files:
            if '.nii.gz' in i_file:
                patient_data = root.split(root_dir)[1]
                patient_ID = patient_data.split(os.sep)[1]
                patient_date = patient_data.split(os.sep)[2]

                file_name = i_file.split('.nii.gz')[0]
                image = sitk.ReadImage(os.path.join(root, i_file), sitk.sitkFloat32)

                for i_i_slice in range(0, image.GetDepth()):
                    out_file_name = os.path.join(out_dir, patient_ID + '_' + patient_date + '__' + file_name + '_' + str(i_i_slice) + '.nii.gz')
                    i_slice = image[:, :, i_i_slice]
                    sitk.WriteImage(i_slice, out_file_name)

    return out_dir


# def rescale_image_intensity(root_dir):
#     for root, dirs, files in os.walk(root_dir):
#         N_files = len(files)
#         ix = 0
#         for i_file in files:
#             if '.nii.gz' in i_file:
#                 ix = ix + 1
#                 print(f'Rescaling intensity... Current image: {ix} / {N_files}')
#                 image = sitk.ReadImage(os.path.join(root, i_file), sitk. sitkFloat32)
#                 image = sitk.RescaleIntensity(image, 0, 1)
#                 sitk.WriteImage(image, os.path.join(root, i_file))

#     return

# replace the original frescaling unction by one that allows multiprocessing since slicing the MRs creates 
# quite many files that have to be accessed individually.
# keep in mind, that, unlike the original function, this function expects all slices to be in the same directory
def rescale_image(image_path):
    if '.nii.gz' in image_path:
        image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        image = sitk.RescaleIntensity(image, 0, 1)
        sitk.WriteImage(image, image_path)
    return

def rescale_image_intensity(root_dir):
    image_list = os.listdir(root_dir)
    path_list = [os.path.join(root_dir, image) for image in image_list]
    with Pool(os.cpu_count()) as p:
        print('Rescaling intensity...')
        for _ in tqdm(p.imap_unordered(rescale_image, path_list), total=len(path_list)):
            pass
    return root_dir


def create_label_file(root_dir, images_4D_file):
    base_dir = os.path.dirname(os.path.normpath(root_dir))
    data_dir = os.path.join(base_dir, 'DATA')

    label_file = os.path.join(data_dir, 'labels.txt')

    images_4D = np.genfromtxt(images_4D_file, dtype='str')

    with open(label_file, 'w') as the_file:
        for root, dirs, files in os.walk(root_dir):
            for i_file in files:
                if '.nii.gz' in i_file:
                    file_name = i_file.split('.nii.gz')[0].split('_')[0:-1]
                    file_name = '_'.join(file_name)
                    if file_name in images_4D:
                        is_4D = '1'
                    else:
                        is_4D = '0'

                    file_location = os.path.join(root, i_file)

                    out_elements = [file_location, '0', is_4D]

                    the_file.write('\t'.join(out_elements) + '\n')

    return label_file
