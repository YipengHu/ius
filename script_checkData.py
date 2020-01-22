
import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt

flag_wsl = True
nSbj = 6
nFrm = 16

if os.name == 'nt':
    home_dir = os.path.expanduser('~')
elif os.name == 'posix':
    if os.uname()[2][-9:]=='Microsoft':
        home_dir = os.path.join('/mnt/c/Users/',os.environ['WINDOWS_USER_NAME'])  # WSL user-provided
    else:
        home_dir = os.environ['HOME']
filename = os.path.join(home_dir, 'Scratch/data/protocol/normalised/protocol_sweep_class_subjects.h5')


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
