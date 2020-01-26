
import tensorflow as tf
import random
import os

import utils2d as utils

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# parameters
validatoin_split = 0.2


idx_model = 1

import sys
parsed_input = sys.argv  # get idx at runtime
if len(parsed_input) == 2:
    idx_model = int(parsed_input[1])

if os.name == 'nt':
    home_dir = os.path.expanduser('~')
elif os.name == 'posix':
    if os.uname()[2][-9:]=='Microsoft':
        home_dir = os.path.join('/mnt/c/Users/',os.environ['WINDOWS_USER_NAME'])  # WSL user-provided
    else:
        home_dir = os.environ['HOME']
filename = os.path.join(home_dir, 'Scratch/data/protocol/normalised/protocol_sweep_class_subjects.h5')

frame_size = tf.keras.utils.HDF5Matrix(filename, '/frame_size').data[()]
frame_size = [int(frame_size[0][0]),int(frame_size[1][0])]
num_classes = tf.keras.utils.HDF5Matrix(filename, '/num_classes').data.value[0][0]

# now get the data using a generator
num_subjects = tf.keras.utils.HDF5Matrix(filename, '/num_subjects').data.value[0][0]
subject_indices = [idx for idx in range(num_subjects)]
random.shuffle(subject_indices) # shuffle once
# split
num_validation = int(num_subjects*validatoin_split)
subject_train, subject_validation = subject_indices[num_validation:], subject_indices[:num_validation]
'''
total_num_frames = 0
for iSbj in subject_indices:
    num_frames = tf.keras.utils.HDF5Matrix(filename, '/subject%06d_num_frames' % iSbj)[0][0]
    total_num_frames += num_frames
'''


# num_frames_per_subject = 1
def data_generator(sub_indices):
    tf.random.shuffle(sub_indices)
    for iSbj in sub_indices:
        num_frames = tf.keras.utils.HDF5Matrix(filename, '/subject%06d_num_frames' % iSbj)[0][0]
        frame_indices = tf.random.shuffle(range(num_frames))
        for idx_frame in frame_indices:  # idx_frame = random.sample(range(num_frames),num_frames_per_subject)[0]
            frame = tf.expand_dims(tf.transpose(tf.cast(tf.keras.utils.HDF5Matrix(filename, '/subject%06d_frame%08d' % (iSbj, idx_frame)), dtype=tf.float32)) / 255.0, axis=0)
            frame = tf.transpose(utils.random_image_transform(frame),[1,2,0])  # data augmentation - plt.imshow(frame[...,0],cmap='gray'), plt.show()
            label = tf.keras.utils.HDF5Matrix(filename, '/subject%06d_label%08d' % (iSbj, idx_frame))[0][0]
            yield (frame, label)


# models
if idx_model == 0:
    model = tf.keras.applications.Xception(
        include_top=True,
        weights=None,
        input_shape=frame_size+[1],
        classes=num_classes)
    print('********** Xception **********')
elif idx_model == 1:
    model = tf.keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        input_shape=frame_size+[1],
        classes=num_classes)
    print('********** ResNet50V2 **********')
elif idx_model == 2:
    model = tf.keras.applications.DenseNet201(
        include_top=True,
        weights=None,
        input_shape=frame_size+[1],
        classes=num_classes)
    print('********** DenseNet201 **********')
elif idx_model == 3:
    model = tf.keras.applications.InceptionV3(
        include_top=True,
        weights=None,
        input_shape=frame_size+[1],
        classes=num_classes)
    print('********** InceptionV3 **********')

# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['SparseCategoricalAccuracy'])


# training
dataset_train = tf.data.Dataset.from_generator(generator=data_generator, args=[subject_train], 
                                               output_types=(tf.float32, tf.int32),
                                               output_shapes=(frame_size+[1], ()))

dataset_val = tf.data.Dataset.from_generator(generator=data_generator, args=[subject_validation], 
                                             output_types=(tf.float32, tf.int32),
                                             output_shapes=(frame_size+[1], ()))

# train_batch = dataset_train.shuffle(buffer_size=1024).batch(128)

model.fit(dataset_train.batch(128), validation_data=dataset_val, epochs=int(1000))
