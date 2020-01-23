
import tensorflow as tf
import random
import os

import utils2d as utils

os.environ["CUDA_VISIBLE_DEVICES"]="0"


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
subject_indices = range(num_subjects)
total_num_frames = 0
for iSbj in subject_indices:
    num_frames = tf.keras.utils.HDF5Matrix(filename, '/subject%06d_num_frames' % iSbj)[0][0]
    total_num_frames += num_frames

'''
images = tf.stack([tf.transpose(tf.cast(tf.keras.utils.HDF5Matrix(
    filename, '/subject%06d_frame%08d' % (0, i)), dtype=tf.float32)) / 255.0 for i in [100,50,75,200,7]], axis=0)
sample_grids = utils.warp_grid(utils.get_reference_grid([images.shape[0]]+frame_size), 
                                utils.random_transform_generator(images.shape[0], corner_scale=0.1))
warped_images = utils.resample_linear(images, sample_grids)
from matplotlib import pyplot as plt
for j in range(images.shape[0]):
    # plt.figure()
    # plt.imshow(images[j,...],cmap='gray')
    plt.figure()
    plt.imshow(warped_images[j,...],cmap='gray')
plt.show()
'''

# num_frames_per_subject = 1
def data_generator():
    for iSbj in subject_indices:
        num_frames = tf.keras.utils.HDF5Matrix(filename, '/subject%06d_num_frames' % iSbj)[0][0]        
        for idx_frame in range(num_frames):  # idx_frame = random.sample(range(num_frames),num_frames_per_subject)[0]
            frame = tf.transpose(tf.cast(tf.keras.utils.HDF5Matrix(filename, '/subject%06d_frame%08d' % (iSbj, idx_frame)), dtype=tf.float32)) / 255.0
            # data augmentation
            warped_grid = utils.warp_grid(utils.get_reference_grid([1]+frame_size), utils.random_transform_generator(1))
            frame = tf.transpose(utils.resample_linear(tf.expand_dims(frame,axis=0), warped_grid),[1,2,0])  # plt.imshow(frame,cmap='gray'), plt.show()
            label = tf.keras.utils.HDF5Matrix(filename, '/subject%06d_label%08d' % (iSbj, idx_frame))[0][0]
            yield (frame, label)

dataset = tf.data.Dataset.from_generator(generator = data_generator,
                                         output_types = (tf.float32, tf.int32),
                                         output_shapes = (frame_size+[1], ()))

# place holder for input image frames
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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['SparseCategoricalAccuracy'])


# training
dataset_batch = dataset.shuffle(buffer_size=1024).batch(total_num_frames)
frame_train, label_train = next(iter(dataset_batch))
model.fit(frame_train, label_train, epochs=int(25000), validation_split=0.2)
