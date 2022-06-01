import os

import numpy as np
from numpy import random

np.random.seed(10086)

trjs = np.load('../../data/geolife_extracted/trjs.npy', allow_pickle=True)
labels = np.load('../../data/geolife_extracted/labels.npy')

n = len(labels)
sub_ratio = 0.5

sub_samples = random.randint(n, size=(int(n*sub_ratio)))

sub_trjs = []
sub_labels = []
for idx in sub_samples:
    sub_trjs.append(trjs[idx])
    sub_labels.append(labels[idx])
print()
if not os.path.exists('../../data/geolife_sub_extracted'):
    os.makedirs('../../data/geolife_sub_extracted')
np.save('../../data/geolife_sub_extracted/trjs.npy', np.array(sub_trjs))
np.save('../../data/geolife_sub_extracted/labels.npy', np.array(sub_labels))