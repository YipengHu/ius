
import tensorflow as tf
import random
import os
# import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"


idx_model = 0

import sys
parsed_input = sys.argv  # get idx at runtime
if len(parsed_input) == 2:
    idx_model = int(parsed_input[1])

if os.name == 'nt':
    home_dir = os.path.expanduser('~')
elif os.name == 'posix':
    if os.uname()[2][-9:]=='Microsoft':
        home_dir = os.path.join('/mnt/c/Users/yhu')  # WSL
    else:
        home_dir = os.environ['HOME']
filename = os.path.join(home_dir, 'Scratch/data/protocol/normalised/protocol_sweep_class_subjects.h5')

frame_size = tf.keras.utils.HDF5Matrix(filename, '/frame_size').data.value
frame_size = [frame_size[0][0],frame_size[1][0]]
num_classes = tf.keras.utils.HDF5Matrix(filename, '/num_classes').data.value[0][0]

# place holder for input image frames
if idx_model == 0:
    model = tf.keras.applications.xception.Xception(
        include_top=True,
        weights=None,
        input_shape=frame_size+[1],
        classes=num_classes)
    print('********** Xception **********')
elif idx_model == 1:
    model = tf.keras.applications.resnet.ResNet50V2(
        include_top=True,
        weights=None,
        input_shape=frame_size+[1],
        classes=num_classes)
    print('********** ResNet50V2 **********')
elif idx_model == 2:
    model = tf.keras.applications.densenet.DenseNet201(
        include_top=True,
        weights=None,
        input_shape=frame_size+[1],
        classes=num_classes)
    print('********** DenseNet201 **********')
elif idx_model == 3:
    model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights=None,
        input_shape=frame_size+[1],
        classes=num_classes)
    print('********** InceptionV3 **********')

# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['SparseCategoricalAccuracy'])


# now get the data using a generator
num_subjects = tf.keras.utils.HDF5Matrix(filename, '/num_subjects').data.value[0][0]
subject_indices = range(num_subjects)
num_frames_per_subject = 1
def data_generator():
    for iSbj in subject_indices:
        dataset = '/subject%06d_num_frames' % iSbj
        num_frames = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
        idx_frame = random.sample(range(num_frames),num_frames_per_subject)[0]
        dataset = '/subject%06d_frame%08d' % (iSbj, idx_frame)
        frame = tf.transpose(tf.keras.utils.HDF5Matrix(filename, dataset)) / 255
        dataset = '/subject%06d_label%08d' % (iSbj, idx_frame)
        label = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
        yield (tf.expand_dims(frame, axis=2), label)

dataset = tf.data.Dataset.from_generator(generator = data_generator,
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size+[1], ()))


# training
dataset_batch = dataset.shuffle(buffer_size=1024).batch(num_subjects)
frame_train, label_train = next(iter(dataset_batch))
model.fit(frame_train, label_train, epochs=int(1e3), validation_split=0.2)
