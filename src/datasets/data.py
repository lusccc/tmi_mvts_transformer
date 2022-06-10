import os
from multiprocessing import cpu_count

import logzero
from logzero import logger
import numpy as np
import pandas as pd


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class FeatureData(object):
    def __init__(self, limit_size=None, config=None):

        # self.set_num_processes(n_proc=n_proc)

        self.config = config
        self.data_name = config['data_name']

        self.all_noise_df, self.all_clean_df,  self.all_masks_df, self.labels_df = self.load()
        '''
        0        1     2         3         4             5     6        7               8 
        delta_t, hour, distance, velocity, acceleration, jerk, heading, heading_change, heading_change_rate
        '''
        use_features = config['motion_features']
        self.all_noise_df = self.all_noise_df[use_features]
        self.all_clean_df = self.all_clean_df[use_features]
        self.all_masks_df = self.all_masks_df[use_features]
        self.all_IDs = self.all_noise_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_noise_df = self.all_noise_df.loc[self.all_IDs]
            self.all_clean_df = self.all_clean_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_noise_df.columns
        self.noise_feature_df = self.all_noise_df
        self.clean_feature_df = self.all_clean_df
        self.masks_df = self.all_masks_df

        self.feature_dfs = [(self.noise_feature_df, 'standardization'),
                            (self.clean_feature_df, 'standardization')]

    def load(self):
        noise_df_path = f"./data/{self.config['data_name']}_features/{self.config['data_name']}_{self.__class__.__name__}_noise_df.pkl"
        clean_df_path = f"./data/{self.config['data_name']}_features/{self.config['data_name']}_{self.__class__.__name__}_clean_df.pkl"
        labels_df_path = f"./data/{self.config['data_name']}_features/{self.config['data_name']}_{self.__class__.__name__}_labels_df.pkl"
        masks_df_path = f"./data/{self.config['data_name']}_features/{self.config['data_name']}_{self.__class__.__name__}_masks_df.pkl"

        if os.path.exists(noise_df_path) and \
                os.path.exists(clean_df_path) and \
                os.path.exists(labels_df_path) and \
                os.path.exists(masks_df_path):
            logger.info(f'read dataframe from file: {noise_df_path}, {clean_df_path}')
            noise_df = pd.read_pickle(noise_df_path)
            clean_df = pd.read_pickle(clean_df_path)
            masks_df = pd.read_pickle(masks_df_path)
            labels_df = pd.read_pickle(labels_df_path)

            lengths = noise_df.groupby(noise_df.index).count()[0].values
            self.max_seq_len = int(np.max(lengths))
        else:
            noise_multi_feature_segs_np = np.load(f'./data/{self.data_name}_features/noise_multi_feature_segs.npy',
                                                  allow_pickle=True)
            noise_multi_feature_seg_labels_np = np.load(
                f'./data/{self.data_name}_features/noise_multi_feature_seg_labels.npy')
            noise_multi_feature_segs = pd.DataFrame(noise_multi_feature_segs_np)
            fs_seg_masks_np = np.load(f'./data/{self.data_name}_features/fs_seg_masks.npy', allow_pickle=True)

            # deprecated
            # fs_seg_masks_np = np.array(
            #     [fs_seg_masks_np for _ in range(noise_multi_feature_segs_np.shape[1])]).T  # duplicate for all feature

            fs_seg_masks = pd.DataFrame(fs_seg_masks_np)
            # ---------- * note noise_multi_feature_seg_labels and
            # clean_multi_feature_seg_labels are same, either will be ok
            labels_df = pd.DataFrame(noise_multi_feature_seg_labels_np)
            clean_multi_feature_segs_np = np.load(f'./data/{self.data_name}_features/clean_multi_feature_segs.npy',
                                                  allow_pickle=True)
            clean_multi_feature_segs = pd.DataFrame(clean_multi_feature_segs_np)

            lengths = noise_multi_feature_segs.applymap(lambda x: len(x)).values
            vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
            if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
                self.max_seq_len = int(np.max(lengths[:, 0]))
                logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
            else:
                self.max_seq_len = lengths[0, 0]

            logger.info(f'{self.__class__.__name__} pd.concat ...')
            # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
            # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
            # sample index (i.e. the same scheme as all datasets in this project)
            noise_df = pd.concat(
                (pd.DataFrame(
                    {col: noise_multi_feature_segs.loc[row, col] for col in noise_multi_feature_segs.columns})
                     .reset_index(drop=True)
                     .set_index(pd.Series(lengths[row, 0] * [row])) for row in
                 range(noise_multi_feature_segs.shape[0])),
                axis=0)
            clean_df = pd.concat(
                (pd.DataFrame(
                    {col: clean_multi_feature_segs.loc[row, col] for col in clean_multi_feature_segs.columns})
                     .reset_index(drop=True)
                     .set_index(pd.Series(lengths[row, 0] * [row])) for row in
                 range(clean_multi_feature_segs.shape[0])),
                axis=0)
            masks_df = pd.concat(
                (pd.DataFrame({col: fs_seg_masks.loc[row, col] for col in fs_seg_masks.columns})
                     .reset_index(drop=True)
                     .set_index(pd.Series(lengths[row, 0] * [row])) for row in range(fs_seg_masks.shape[0])),
                axis=0)
            logger.info(f'save to file: {noise_df_path}, {clean_df_path}')
            noise_df.to_pickle(noise_df_path)
            clean_df.to_pickle(clean_df_path)
            masks_df.to_pickle(masks_df_path)
            labels_df.to_pickle(labels_df_path)

        if 'classification' in self.config['task']:
            labels = pd.Series(labels_df.values.reshape(-1), dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes,
                                     dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        return noise_df, clean_df, masks_df, labels_df


class TrajectoryData(object):

    def __init__(self, limit_size=None, config=None) -> None:
        self.config = config
        self.data_name = config['data_name']

        self.all_noise_df, self.all_clean_df, self.all_masks_df, self.labels_df = self.load()
        self.all_IDs = self.all_noise_df.index.unique()

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_noise_df = self.all_noise_df.loc[self.all_IDs]
            self.all_clean_df = self.all_clean_df.loc[self.all_IDs]
            self.all_masks_df = self.all_masks_df.loc[self.all_IDs]

        # use all features, here feature just mean lat and lon
        self.feature_names = self.all_noise_df.columns
        self.noise_feature_df = self.all_noise_df
        self.clean_feature_df = self.all_clean_df
        self.masks_df = self.all_masks_df

        self.feature_dfs = [(self.noise_feature_df, 'standardization'),
                            (self.clean_feature_df, 'standardization')]

    def load(self):
        noise_df_path = f"./data/{self.config['data_name']}_features/{self.config['data_name']}_{self.__class__.__name__}_noise_df.pkl"
        clean_df_path = f"./data/{self.config['data_name']}_features/{self.config['data_name']}_{self.__class__.__name__}_clean_df.pkl"
        masks_df_path = f"./data/{self.config['data_name']}_features/{self.config['data_name']}_{self.__class__.__name__}_masks_df.pkl"
        labels_df_path = f"./data/{self.config['data_name']}_features/{self.config['data_name']}_{self.__class__.__name__}_labels_df.pkl"
        if os.path.exists(noise_df_path) and \
                os.path.exists(clean_df_path) and \
                os.path.exists(masks_df_path) and \
                os.path.exists(labels_df_path):
            logger.info(f'read dataframe from file: {noise_df_path}, {clean_df_path}')
            noise_df = pd.read_pickle(noise_df_path)
            clean_df = pd.read_pickle(clean_df_path)
            masks_df = pd.read_pickle(masks_df_path)
            labels_df = pd.read_pickle(labels_df_path)

            lengths = noise_df.groupby(noise_df.index).agg(lambda x: x.nunique())[0].values
            self.max_seq_len = int(np.max(lengths))
        else:
            noise_trj_segs_np = np.load(f'./data/{self.data_name}_features/noise_trj_segs.npy', allow_pickle=True)
            clean_trj_segs_np = np.load(f'./data/{self.data_name}_features/clean_trj_segs.npy', allow_pickle=True)
            trj_seg_masks_np = np.load(f'./data/{self.data_name}_features/trj_seg_masks.npy', allow_pickle=True)

            # duplicate for lon and lat, note it is used for the condition of
            # `generate a mask seg by considering lat and lon SIMULTANEOUSLY`
            # trj_seg_masks_np = np.array([trj_seg_masks_np, trj_seg_masks_np]).T

            noise_trj_segs = pd.DataFrame(noise_trj_segs_np)
            clean_trj_segs = pd.DataFrame(clean_trj_segs_np)
            trj_seg_masks = pd.DataFrame(trj_seg_masks_np)
            # ----------* load label for WeightedRandomSampler in main.py, note noise_multi_feature_seg_labels and
            # clean_multi_feature_seg_labels are same, either will be ok
            noise_multi_feature_seg_labels_np = np.load(
                f'./data/{self.data_name}_features/noise_multi_feature_seg_labels.npy')
            labels_df = pd.DataFrame(noise_multi_feature_seg_labels_np)

            lengths = noise_trj_segs.applymap(lambda x: len(x)).values
            vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
            if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
                self.max_seq_len = int(np.max(lengths[:, 0]))
                logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
            else:
                self.max_seq_len = lengths[0, 0]

            noise_df = pd.concat(
                (pd.DataFrame({col: noise_trj_segs.loc[row, col] for col in noise_trj_segs.columns})
                     .reset_index(drop=True)
                     .set_index(pd.Series(lengths[row, 0] * [row])) for row in range(noise_trj_segs.shape[0])),
                axis=0)
            clean_df = pd.concat(
                (pd.DataFrame({col: clean_trj_segs.loc[row, col] for col in clean_trj_segs.columns})
                     .reset_index(drop=True)
                     .set_index(pd.Series(lengths[row, 0] * [row])) for row in range(clean_trj_segs.shape[0])),
                axis=0)
            masks_df = pd.concat(
                (pd.DataFrame({col: trj_seg_masks.loc[row, col] for col in trj_seg_masks.columns})
                     .reset_index(drop=True)
                     .set_index(pd.Series(lengths[row, 0] * [row])) for row in range(trj_seg_masks.shape[0])),
                axis=0)
            noise_df.to_pickle(noise_df_path)
            clean_df.to_pickle(clean_df_path)
            masks_df.to_pickle(masks_df_path)
            labels_df.to_pickle(labels_df_path)

        if 'classification' in self.config['task']:
            labels = pd.Series(labels_df.values.reshape(-1), dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes,
                                     dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        return noise_df, clean_df, masks_df, labels_df


class TrajectoryWithFeatureData(object):
    def __init__(self, limit_size=None, config=None) -> None:
        self.config = config
        self.data_name = config['data_name']
        self.feature_data = FeatureData(limit_size, config)
        self.trajectory_data = TrajectoryData(limit_size, config)
        self.all_IDs = self.feature_data.all_IDs
        self.feature_dfs = self.trajectory_data.feature_dfs + self.feature_data.feature_dfs
        self.labels_df = self.feature_data.labels_df


data_factory = {
    'trajectory': TrajectoryData,
    'feature': FeatureData,
    'trajectory_with_feature': TrajectoryWithFeatureData
}

if __name__ == '__main__':
    print()
