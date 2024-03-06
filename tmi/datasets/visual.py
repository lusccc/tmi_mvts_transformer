import numpy as np
import matplotlib.pyplot as plt

dataset = 'SHL'

shl_n_noise_point_dist = [  639, 31586, 58455, 22270, 23706] # n_removed_points in trajectory_feature_calculation.py


def count_number_each_class(dataset):
    x = np.load(f'../../data/{dataset}_features/clean_multi_feature_segs.npy', allow_pickle=True)
    y = np.load(f'../../data/{dataset}_features/clean_multi_feature_seg_labels.npy', )

    for x_sample, y_sample in zip(x, y):


    unique, n_each = np.unique(y, return_counts=True)
    return n_each


plt.figure()
x = ['walk', 'bike', 'bus', 'driving', 'train']
# create an index for each tick position
xi = np.array(list(range(len(x))))

axes = plt.gca()
# axes.set_xlim([min(x),max(x)])
# axes.set_ylim([min(y) - 0.003, max(y) + .001])
axes.set_ylim([300,1500])
# y1 = count_number_each_class('geolife')
y2 = count_number_each_class('SHL')
# plot the index for the x-values
# bars1 = plt.bar(xi - 0.2, y1, width=0.4, label='Geolife' )
bars2 = plt.bar(xi, y2, width=0.6, label='SHL')
# for bar in bars1:
#     yval = bar.get_height()
#     plt.text(bar.get_x()+.025, yval + 300, '%d' % yval, va='top', fontsize=8, )
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x()+.110, yval + 60, '%d' % yval, va='top', fontsize=8, )
plt.xlabel('Transportation Mode')
plt.ylabel('Number of Segments')
plt.xticks(xi, x, fontsize=8, )
# plt.legend()
plt.show()
