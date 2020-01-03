
import os
import tensorflow as tf
from matplotlib import pyplot as plt


norm_folder = 'Scratch/data/protocol/normalised'
if os.name == 'nt':
    home_dir = os.path.expanduser('~')
else:
    home_dir = os.environ['HOME']

filename = os.path.join(home_dir, norm_folder, 'protocol_sweep_class_subjects.h5')
group_name = '/subject%06d_frame%08d' % (20, 10)
frame = tf.keras.utils.HDF5Matrix(filename, group_name)

plt.figure()
plt.imshow(frame, cmap='gray')
plt.show()
