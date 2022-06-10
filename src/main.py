"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import logzero
import numpy as np
from logzero import logger
from torch.backends import cudnn
from torch.profiler import tensorboard_trace_handler
import os
import sys
import time
import pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Project modules
from datasets import dataset
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.models import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer
from training_tools import EarlyStopping

# to address Too many open files Pin memory thread exited unexpectedly:
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger.info("Loading packages ...")


def main(config):
    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = 'cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu'
    if device != 'cuda':
        logger.error("no cuda!! ")
        # exit(-1)
        device = torch.device(device)
    else:
        device = torch.device(device)
        logger.info("Using device: {}".format(device))
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # ***********  Build data ***********
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']]
    my_data = data_class(limit_size=config['limit_size'], config=config)
    # feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    if 'classification' in config['task']:
        # validation_method = 'StratifiedShuffleSplit'
        validation_method = 'ShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    # Split dataset
    test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    # load test IDs directly from file, if available, otherwise use `test_set_ratio`.
    if config['test_from']:
        test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
        try:
            test_indices = [int(ind) for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
        logger.info("Loaded {} test IDs from file: '{}'".format(len(test_indices), config['test_from']))

    # Note: currently a validation set must exist,
    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                 validation_method=validation_method,
                                                                 n_splits=1,
                                                                 validation_ratio=config['val_ratio'],
                                                                 test_set_ratio=config['test_ratio'],
                                                                 # used only if test_indices not explicitly specified
                                                                 test_indices=test_indices,
                                                                 random_seed=10086,
                                                                 labels=labels)
        train_indices = train_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
        val_indices = val_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices),
                       'test_indices': list(test_indices)}, f, indent=4)

    # *********** Pre-process features ***********
    for df, normalization in my_data.feature_dfs:
        normalizer = Normalizer(normalization)
        df.loc[train_indices] = normalizer.normalize(df.loc[train_indices])
        if len(val_indices):
            df.loc[val_indices] = normalizer.normalize(df.loc[val_indices])
        if len(test_indices):
            df.loc[test_indices] = normalizer.normalize(df.loc[test_indices])

    # *********** Create model ***********
    logger.info("Creating model ...")
    model = model_factory(config, my_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if 'output_layer' in name:
                logger.info(f'set layer {name} requires_grad = True')
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer
    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if config['task'] == 'dual_branch_classification':
        if config['load_trajectory_branch']:
            model.trajectory_branch, _, __ = utils.load_model(model.trajectory_branch,
                                                              config['load_trajectory_branch'], optimizer,
                                                              config['resume'],
                                                              config['change_output'],
                                                              config['lr'],
                                                              config['lr_step'],
                                                              config['lr_factor'])
        if config['load_feature_branch']:
            model.feature_branch, _, __ = utils.load_model(model.feature_branch,
                                                           config['load_feature_branch'], optimizer,
                                                           config['resume'],
                                                           config['change_output'],
                                                           config['lr'],
                                                           config['lr_step'],
                                                           config['lr_factor'])

    if config['load_model']:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)
    loss_module = get_loss_module(config)

    # *********** Only evaluate ***********
    def evaluate_test():
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, ))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                      print_interval=config['print_interval'], console=config['console'])
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            if v is not None:
                print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)

    if config['test_only'] == 'testset':  # Only evaluate and skip training
        evaluate_test()
        return

    # *********** Initialize data generators ***********
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, val_indices)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=collate_fn)

    # not used !!!! construct WeightedRandomSampler,
    # see https://medium.com/analytics-vidhya/augment-your-data-easily-with-pytorch-313f5808fc8b
    train_dataset = dataset_class(my_data, train_indices)
    # train_label_unique, counts = np.unique(train_dataset.labels_df, return_counts=True)
    # class_weights = [sum(counts) / c for c in counts]
    # sample_weights = [class_weights[e] for e in train_dataset.labels_df[0]]
    # sampler = WeightedRandomSampler(sample_weights, len(train_dataset.labels_df[0]))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              # sampler=sampler,
                              collate_fn=collate_fn)

    # *********** TRICKS **********
    logger.info('data loader to list ...')
    # note will be automatically shuffled if the DataLoader with  shuffle=True
    val_loader = list(val_loader)
    train_loader = list(train_loader)

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                           print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                 print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    # dataloader time consumption test
    if False:
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=2,
                    active=6,
                    repeat=1),
                on_trace_ready=tensorboard_trace_handler('./'),
                with_stack=True
        ) as profiler:
            t_sum = 0
            last_time = time.time()
            for i, data in enumerate(train_loader):
                t = time.time()
                t_diff = t - last_time
                t_sum += t_diff
                last_time = t
                logger.info(f't_diff: {t_diff}')
                profiler.step()
            logger.info(f't_sum: {t_sum}')
            exit(0)

    # *********** Train record ***********
    best_value = 1e16 \
        if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # *********** Evaluate on validation before training ***********
    metrics_names = None
    if config['val_ratio'] > 0:
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch=0)
        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))
    else:
        if 'classification' in config['task']:
            metrics_names = ('epoch', 'loss', 'accuracy', 'precision')
        else:
            metrics_names = ('epoch', 'loss')

    # *********** Start training ***********
    logger.info('Starting training...')
    early_stop = False
    early_stopping = EarlyStopping(round(config['patience']/config['val_interval']), verbose=True)
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info(
            "Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # evaluate if first or last epoch or at specified interval
        if config['val_ratio'] > 0 and \
                ((epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0)):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

            if early_stopping(aggr_metrics_val['loss']).early_stop:
                early_stop = True
                logger.warn('early stopping reached')

        utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        if early_stop:
            logger.warn('stopping training...')
            break

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model,
                             optimizer)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # *********** Results ***********
    # Export evolution of metrics over epochs
    # if 'classification' in config['task']:
    #     model =
    #     evaluate_test()
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, str(best_metrics)))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    return best_value


if __name__ == '__main__':
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    dataset.config = config
    cudnn.benchmark = True
    #  paper: Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision
    torch.manual_seed(3407)
    main(config)
