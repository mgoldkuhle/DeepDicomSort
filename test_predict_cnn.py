# testing multiprocessing of the prediction. seems to perform worse than single threaded. 

from tensorflow.keras.models import load_model
import numpy as np
import Tools.data_IO as data_IO
import tensorflow as tf
import yaml
import os
from multiprocessing import Pool
from tqdm import tqdm

model_file = './Trained_Models/model_all_brain_tumor_data.hdf5'
batch_size = 1

with open('./config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

test_label_file = cfg['testing']['test_label_file']
output_folder = cfg['testing']['output_folder']
x_image_size = cfg['data_preparation']['image_size_x']
y_image_size = cfg['data_preparation']['image_size_y']

model_name = os.path.basename(os.path.normpath(model_file)).split('.hdf5')[0]
out_file = os.path.join(output_folder, 'Predictions_' + model_name + '.csv')


def load_labels(label_file):
    labels = np.genfromtxt(label_file, dtype='str', delimiter='\t')
    label_IDs = labels[:, 0]
    label_IDs = np.asarray(label_IDs)
    label_values = labels[:, 1].astype(np.int)
    extra_inputs = labels[:, 2:].astype(np.float)
    np.round(extra_inputs, 2)

    N_classes = len(np.unique(label_values))

    # Make sure that minimum of labels is 0
    label_values = label_values - np.min(label_values)

    return label_IDs, label_values, N_classes, extra_inputs


test_image_IDs, test_image_labels, _, extra_inputs = load_labels(test_label_file)


optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)


NiftiGenerator_test = data_IO.NiftiGenerator2D_ExtraInput(batch_size,
                                                           test_image_IDs,
                                                           test_image_labels,
                                                           [x_image_size, y_image_size],
                                                           extra_inputs)



def predict_image(args):
    i_file, i_label, i_extra_input = args

    model = load_model(model_file)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['categorical_accuracy']
    )

    image = NiftiGenerator_test.get_single_image(i_file)

    supplied_extra_input = np.zeros([1, 1])
    supplied_extra_input[0, :] = i_extra_input
    prediction = model.predict([image, supplied_extra_input])
    return i_file, np.argmax(prediction) + 1, i_label

# Prepare arguments for predict_image
args = [(i_file, i_label, i_extra_input, ) for i_file, i_label, i_extra_input in zip(test_image_IDs, test_image_labels, extra_inputs)]

# Create a pool of worker processes. The feasible number of processes depend on your machine. 
with Pool(4) as p:
    # Open the output file
    with open(out_file, 'w') as the_file:
        # Apply the function to each set of arguments in parallel
        print('Predicting images...')
        for i_file, prediction, i_label in tqdm(p.imap_unordered(predict_image, args), total=len(args)):
            # Write the result immediately
            the_file.write(f"{i_file}\t{prediction}\t{i_label}\n")