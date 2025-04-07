import argparse


class Options(object):

    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description='Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.')

        ## Run from config file
        self.parser.add_argument('--config', dest='config_filepath',
                                 help='Configuration .json file (optional). Overwrites existing command-line args!')

        ## Run from command-line arguments
        # I/O
        self.parser.add_argument('--output_dir', default='./output',
                                 help='Root output directory. Must exist. Time-stamped directories will be created inside.')
        self.parser.add_argument('--load_model',
                                 help='Path to pre-trained model.')
        self.parser.add_argument('--resume', action='store_true',
                                 help='If set, will load `starting_epoch` and state of optimizer, besides model weights.')
        self.parser.add_argument('--change_output', action='store_true',
                                 help='Whether the loaded model will be fine-tuned on a different task (necessitating a different output layer)')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='If set, will save model weights (and optimizer state) for every epoch; otherwise just latest')
        self.parser.add_argument('--experiment_name', default='',
                                 help='A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp')
        self.parser.add_argument('--records_file', default='./records.xls',
                                 help='Excel file keeping all records of experiments')
        # System
        self.parser.add_argument('--console', action='store_true',
                                 help="Optimize printout for console output; otherwise for file")
        self.parser.add_argument('--print_interval', type=int, default=1,
                                 help='Print batch info every this many batches')
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='GPU index, -1 for CPU')
        self.parser.add_argument('--n_proc', type=int, default=-1,
                                 help='Number of processes for data loading/preprocessing. By default, equals num. of available cores.')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed',
                                 help='Seed used for splitting sets. None by default, set to an integer for reproducibility')
        # Dataset
        self.parser.add_argument('--limit_size', type=float, default=None,
                                 help="Limit  dataset to specified smaller random sample, e.g. for rapid debugging purposes. "
                                      "If in [0,1], it will be interpreted as a proportion of the dataset, "
                                      "otherwise as an integer absolute number of samples")
        self.parser.add_argument('--test_only', choices={'testset', 'fold_transduction'},
                                 help='If set, no training will take place; instead, trained model will be loaded and evaluated on test set')
        self.parser.add_argument('--data_name', type=str, help='dataset name')
        self.parser.add_argument('--data_class', type=str,
                                 choices={'feature', 'trajectory', 'trajectory_with_feature'},
                                 help="Which type of data should be processed.")
        # already split in s2 step!
        # self.parser.add_argument('--test_ratio', type=float, default=0,
        #                          help="Set aside this proportion of the dataset as a test set")
        self.parser.add_argument('--val_ratio', type=float, default=0.2,
                                 help="Proportion of the dataset to be used as a validation set")

        # already specified in data.py
        # self.parser.add_argument('--normalization',
        #                          choices={'standardization', 'minmax', 'per_sample_std', 'per_sample_minmax'},
        #                          default='standardization',
        #                          help='If specified, will apply normalization on the input features of a dataset.')

        # Training process
        self.parser.add_argument('--task',
                                 help=("Training objective/task: imputation of masked values,\n"
                                       "                          transduction of features to other features,\n"
                                       "                          classification of entire time series,\n"
                                       "                          regression of scalar(s) for entire time series"))
        self.parser.add_argument('--exclude_feats', type=str, default=None,
                                 help='Imputation: Comma separated string of indices corresponding to features to be excluded from masking')

        self.parser.add_argument('--epochs', type=int, default=400,
                                 help='Number of training epochs')
        self.parser.add_argument('--val_interval', type=int, default=2,
                                 help='Evaluate on validation set every this many epochs. Must be >= 1.')
        self.parser.add_argument('--optimizer', choices={"Adam", "RAdam"}, default="Adam", help="Optimizer")
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='learning rate (default holds for batch size 64)')
        self.parser.add_argument('--lr_step', type=str, default='1000000',
                                 help='Comma separated string of epochs when to reduce learning rate by a factor of 10.'
                                      ' The default is a large value, meaning that the learning rate will not change.')
        self.parser.add_argument('--lr_factor', type=str, default='0.1',
                                 help=("Comma separated string of multiplicative factors to be applied to lr "
                                       "at corresponding steps specified in `lr_step`. If a single value is provided, "
                                       "it will be replicated to match the number of steps in `lr_step`."))
        self.parser.add_argument('--batch_size', type=int, default=64,
                                 help='Training batch size')
        self.parser.add_argument('--l2_reg', type=float, default=0,
                                 help='L2 weight regularization parameter')
        self.parser.add_argument('--global_reg', action='store_true',
                                 help='If set, L2 regularization will be applied to all weights instead of only the output layer')
        self.parser.add_argument('--freeze', action='store_true',
                                 help='If set, freezes all layer parameters except for the output layer. Also removes dropout except before the output layer')

        # Model
        self.parser.add_argument('--model', choices={"transformer", "LINEAR"}, default="transformer",
                                 help="Model class")
        self.parser.add_argument('--max_seq_len', type=int,
                                 help="""Maximum input sequence length. Determines size of transformer layers.
                                 If not provided, then the value defined inside the data class will be used.""")
        self.parser.add_argument('--data_window_len', type=int,
                                 help="""Used instead of the `max_seq_len`, when the data samples must be
                                 segmented into windows. Determines maximum input sequence length 
                                 (size of transformer layers).""")
        self.parser.add_argument('--d_model', type=int, default=64,
                                 help='Internal dimension of transformer embeddings')
        self.parser.add_argument('--dim_feedforward', type=int, default=256,
                                 help='Dimension of dense feedforward part of transformer layer')
        self.parser.add_argument('--num_heads', type=int, default=8,
                                 help='Number of multi-headed attention heads')
        self.parser.add_argument('--num_layers', type=int, default=3,
                                 help='Number of transformer encoder layers (blocks)')
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                 help='Dropout applied to most transformer encoder layers')
        self.parser.add_argument('--pos_encoding', choices={'fixed', 'learnable'}, default='fixed',
                                 help='Internal dimension of transformer embeddings')
        self.parser.add_argument('--activation', choices={'relu', 'gelu'}, default='gelu',
                                 help='Activation to be used in transformer encoder')
        self.parser.add_argument('--normalization_layer', choices={'BatchNorm', 'LayerNorm'}, default='BatchNorm',
                                 help='Normalization layer to be used internally in transformer encoder')
        self.parser.add_argument('--trajectory_branch_hyperparams', type=str, default=None)
        self.parser.add_argument('--feature_branch_hyperparams', type=str, default=None)
        self.parser.add_argument('--load_trajectory_branch', type=str, default=None)
        self.parser.add_argument('--load_feature_branch', type=str, default=None)
        self.parser.add_argument('--input_type', default='50%noise')
        self.parser.add_argument('--disable_mask', action='store_true', )
        '''
        0        1     2         3         4             5     6        7               8 
        delta_t, hour, distance, velocity, acceleration, jerk, heading, heading_change, heading_change_rate
        '''
        self.parser.add_argument('--motion_features', type=str, default='3,4,5,8')
        self.parser.add_argument('--patience', type=int, default=60)
        self.parser.add_argument('--emb_size', type=int, default=64)
        self.parser.add_argument('--class_names', type=str, default="Walk,Bike,Bus,Car,Train",
                                  help='逗号分隔的类别名称列表，按索引顺序对应标签值')

    def parse(self):

        args = self.parser.parse_args()

        args.lr_step = [int(i) for i in args.lr_step.split(',')]
        args.lr_factor = [float(i) for i in args.lr_factor.split(',')]
        if (len(args.lr_step) > 1) and (len(args.lr_factor) == 1):
            args.lr_factor = len(args.lr_step) * args.lr_factor  # replicate
        assert len(args.lr_step) == len(
            args.lr_factor), "You must specify as many values in `lr_step` as in `lr_factors`"

        if args.exclude_feats is not None:
            args.exclude_feats = [int(i) for i in args.exclude_feats.split(',')]

        args.motion_features = [int(item) for item in args.motion_features.split(',')]
        
        args.key_metric = 'accuracy' if 'classification' in args.task else 'loss'
        
        # 处理类别名称
        if args.class_names is not None:
            args.class_names = args.class_names.split(',')
        
        return args
