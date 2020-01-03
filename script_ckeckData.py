
import os
import tensorflow as tf
from matplotlib import pyplot as plt


norm_folder = '/mnt/c/Users/yhu/Scratch/data/protocol/normalised'
if os.name == 'nt':
    home_dir = os.path.expanduser('~')
else:
    home_dir = os.environ['HOME']

filename = os.path.join(home_dir, norm_folder, 'protocol_sweep_class_subjects.h5')

idx_subject = [20, 50, 150, 200]
idx_frame = [0, 15, 20]
nSbj = len(idx_subject)
nFrm = len(idx_frame)
plt.figure()
for iSbj in range(nSbj):
    for iFrm in range(nFrm):
        group_name = '/subject%06d_frame%08d' % (idx_subject[iSbj], idx_frame[iFrm])
        frame = tf.keras.utils.HDF5Matrix(filename, group_name)
        plt.subplot(nSbj, nFrm, iSbj*nFrm+iFrm+1)
        plt.imshow(frame, cmap='gray')
plt.show()
