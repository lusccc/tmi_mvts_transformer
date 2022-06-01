# import packages
import numpy as np
import pandas as pd

from data_test.change_point_detection_data_test import generate_mask_using_CPD

trj = pd.read_csv('./seg.csv', header=None)
# generate_mask_using_CPD(trj.values)


fs_masks = np.load('../data/SHL_features/fs_seg_masks.npy', allow_pickle=True)
trj_masks = np.load('../data/SHL_features/trj_seg_masks.npy', allow_pickle=True)
print()