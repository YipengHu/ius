
import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt

flag_wsl = True

norm_folder = 'Scratch/data/protocol/normalised'
if os.name == 'nt':
    home_dir = os.path.expanduser('~')
elif os.name == 'posix':
    home_dir = os.environ['HOME']
if flag_wsl: 
    home_dir = os.path.join('/mnt/c/Users/yhu')  # WSL

filename = os.path.join(home_dir, norm_folder, 'protocol_sweep_class_subjects.h5')
nSbj = 6
nFrm = 8

# generate 5 random subjects
num_subjects = tf.keras.utils.HDF5Matrix(filename, '/num_subjects').data.value[0][0]
idx_subject = random.sample(range(num_subjects),nSbj)

plt.figure()
for iSbj in range(nSbj):
    dataset = '/subject%06d_num_frames' % (idx_subject[iSbj])
    num_frames = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]
    idx_frame = random.sample(range(num_frames),nFrm)
    for iFrm in range(nFrm):
        dataset = '/subject%06d_frame%08d' % (idx_subject[iSbj], idx_frame[iFrm])
        frame = tf.transpose(tf.keras.utils.HDF5Matrix(filename, dataset))
        dataset = '/subject%06d_label%08d' % (idx_subject[iSbj], idx_frame[iFrm])
        label = tf.keras.utils.HDF5Matrix(filename, dataset)[0][0]

        axs = plt.subplot(nSbj, nFrm, iSbj*nFrm+iFrm+1)
        axs.set_title('S{}, F{}, C{}'.format(idx_subject[iSbj], idx_frame[iFrm], label))
        axs.imshow(frame, cmap='gray')
        axs.axis('off')
plt.show()
