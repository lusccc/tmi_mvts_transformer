# coding=utf-8
import argparse
import os.path

import numpy as np
import pandas as pd
from logzero import logger
from sklearn.utils import shuffle


def process_labels(labels_file_path, locations_file_path):
    sampled_labels = pd.read_csv(labels_file_path, header=None, sep=' ', usecols=[0, 1])
    sampled_labels.columns = ['timestamp', 'label']
    sampled_labels['timestamp'] = sampled_labels['timestamp'].apply(lambda x: x / 1000.)  # seconds

    # https://stackoverflow.com/questions/48997350/pandas-dataframe-groupby-for-separate-groups-of-same-value
    sampled_labels['marker'] = (sampled_labels['label'] != sampled_labels['label'].shift()).cumsum()
    labels_time_range = sampled_labels.groupby('marker').agg({'label': 'first', 'timestamp': lambda x: list(x)})

    gps_records = pd.read_csv(locations_file_path, header=None, sep=' ', usecols=[0, 4, 5])
    gps_records.columns = ['timestamp', 'lat', 'lon']
    gps_records['timestamp'] = gps_records['timestamp'].apply(lambda x: x / 1000.)

    """
    SHL label:
    Null=0, Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7, Subway=8
    
    Geolife label:
    modes: {'walk': 0, 'bike': 1, 'bus': 2, 'car': 3, 'subway': 4, 'train': 4, 'airplane': 6, 'boat': 7, 'run': 8, 'motorcycle': 9, 'taxi': 3}
    modes to use: [0, 1, 2, 3, 4]
    """

    for idx, label_detail in labels_time_range.iterrows():
        label = label_detail['label']
        if label not in use_modes:
            continue
        if len(use_modes) == 5:
            ' !!!!make the label index of modes SAME as geolife!!!! geolife use 5 classification'
            if label == 2:
                label = 0
            # if label == 3:
            #     label = 0
            if label == 4:
                label = 1
            if label == 5:
                label = 3
            if label == 6:
                label = 2
            if label == 7 or label == 8:  # merge train&subway
                label = 4
        st = label_detail['timestamp'][0]
        et = label_detail['timestamp'][-1]
        trj = gps_records[(gps_records['timestamp'] >= st) & (gps_records['timestamp'] <= et)]
        trj = trj[['timestamp', 'lat', 'lon']].values
        trjs.append(trj)
        trjs_labels.append(label)


def read_all_folders(path):
    folders = os.listdir(path)
    n = len(folders)
    for i, f in enumerate(folders):
        logger.info(f'{i + 1}/{n}')
        labels_file_path = os.path.join(path, f, 'Label.txt')
        locations_file_path = os.path.join(path, f, 'Hips_Location.txt')
        if os.path.exists(labels_file_path) and os.path.exists(locations_file_path):
            process_labels(labels_file_path, locations_file_path)
        else:
            logger.info(f'file not exist for {labels_file_path} or {locations_file_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRJ_EXT')
    '''
     mode index from SHL is differ from geolife, will be transform to be same as geolife
    '''
    parser.add_argument('--use_modes', type=str, default='2,4,6,5,7') # TODO 8 actually not used!
    parser.add_argument('--data_dir', type=str, default='/mnt/f/tmi-code/DATASET/SHLDataset_User1Hips_v1/release/User1')
    parser.add_argument('--save_dir', type=str, default='../../data/SHL_extracted/')

    args = parser.parse_args()

    # Null=0, Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7, Subway=8
    # 7,8 are merged to 7
    use_modes = [int(item) for item in args.use_modes.split(',')]
    logger.info('use_modes:', use_modes)

    trjs = []
    trjs_labels = []
    read_all_folders(args.data_dir)
    trjs = np.array(trjs, dtype=object)
    labels = np.array(trjs_labels)
    trjs, labels = shuffle(trjs, labels, random_state=10086)  # note: shuffles here !!
    # trjs = trjs[:200]
    # labels = labels[:200]

    logger.info('saving files...')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    np.save(f'{args.save_dir}trjs.npy', trjs)
    np.save(f'{args.save_dir}labels.npy', labels)

    #  --use_modes "1,2,3,4,6,5,7" --data_dir /mnt/e/DATASET/SHLDataset_User1Hips_v1/release/User1 --save_dir ./data/SHL_7_extracted/