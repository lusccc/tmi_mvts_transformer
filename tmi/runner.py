import json
import os
import pickle
import random
import string
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime

import logzero
import numpy as np
import sklearn
import torch
from logzero import logger
from torch import Tensor
from torch.utils.data import DataLoader

from tmi.datasets.dataset import DenoisingDataset, collate_denoising_unsuperv, collate_generic_superv, \
    ImputationDataset, collate_imputation_unsuperv, DualBranchClassificationDataset, collate_dual_branch_superv, \
    GenericClassificationDataset, DenoisingImputationDataset
from tmi.models.loss import l2_reg_loss, mask_length_regularization_loss
from tmi.models.models import DualTSTransformerEncoderClassifier
from tmi.utils import utils, analysis

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config['task']

    if task == "denoising_pretrain":
        return DenoisingDataset, collate_denoising_unsuperv, UnsupervisedRunner
    if task == "imputation_pretrain":
        return ImputationDataset, collate_imputation_unsuperv, UnsupervisedRunner
    if task == "denoising_imputation_pretrain":
        return DenoisingImputationDataset, collate_imputation_unsuperv, UnsupervisedRunner
    if task in ['dual_branch_classification', 'dual_branch_classification_from_scratch']:
        return DualBranchClassificationDataset, collate_dual_branch_superv, SupervisedRunner
    if task in ['feature_branch_classification', 'feature_branch_classification_from_scratch',
                'trajectory_branch_classification',
                'trajectory_branch_classification_from_scratch', ]:
        return GenericClassificationDataset, collate_generic_superv, SupervisedRunner
    if "cnn_classification" in task or "lstm_classification" in task:
        return GenericClassificationDataset, collate_generic_superv, SupervisedRunner
    if task == 'ml_classification':
        return GenericClassificationDataset, collate_generic_superv, SupervisedRunner
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    # 如果输出目录不存在则创建
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # output_dir = os.path.join(output_dir, config['experiment_name'])
    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    # if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
    #     rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
    #     output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    # config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    utils.create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])
    logzero.logfile(os.path.join(output_dir, 'output.log'), backupCount=3)

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def evaluate(evaluator):
    """NOT USED! Perform a single, one-off evaluation on an evaluator object (initialized with a dataset)"""

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = evaluator.evaluate(epoch_num=None, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    print()
    print_str = 'Evaluation Summary: '
    for k, v in aggr_metrics.items():
        if v is not None:
            print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)
    logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    return aggr_metrics, per_batch


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    # avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader)
    logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = 'Epoch {} Validation Summary: '.format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        # np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value


def check_progress(epoch):
    if epoch in [100, 140, 160, 220, 280, 340]:
        return True
    else:
        return False


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, 
                 l2_reg=None, exp_config=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg

        self.exp_config = exp_config
        self.print_interval = self.exp_config['print_interval']
        self.printer = utils.Printer(console=self.exp_config['console'])
        self.disable_mask = self.exp_config['disable_mask']

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.dataloader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):
            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)

            # 将target_masks作为feature_masks传递给模型
            predictions = self.model(X.to(self.device), padding_masks, 
                                     None if self.disable_mask else target_masks)  # (batch_size, padded_length, feat_dim)

            # 将padding_masks传递给损失函数，以便DenoisingImputationLoss可以正确处理
            loss = self.loss_module(predictions, targets, target_masks, padding_masks)
            
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # 添加mask_length正则化，鼓励mask_length_factor探索更广的值域
            if hasattr(self.model, 'mask_length_factor'):
                mask_reg_loss = mask_length_regularization_loss(self.model, center_value=0.5, strength=0.005)
                total_loss = total_loss + mask_reg_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # 使用梯度裁剪提高训练稳定性
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        
        # 记录当前学习到的信号增强值
        if not self.disable_mask:
            if hasattr(self.model, 'missing_signal_strength'):
                signal_value = self.model.missing_signal_strength.item()
                self.epoch_metrics['signal_strength'] = signal_value
                logger.info(f"当前学习到的信号增强值: {signal_value:.6f}")
                
                # 记录当前学习到的mask长度
                if hasattr(self.model, 'mask_length_factor'):
                    factor_value = self.model.mask_length_factor.item()
                    mask_length = 1 + 9 * torch.sigmoid(torch.tensor(factor_value)).item()  # 更新为1-10范围
                    self.epoch_metrics['mask_length'] = mask_length
                    self.epoch_metrics['mask_length_factor'] = factor_value
                    # 计算上一个epoch的值（如果存在）
                    prev_factor = getattr(self, 'prev_mask_factor', factor_value)
                    delta = factor_value - prev_factor
                    setattr(self, 'prev_mask_factor', factor_value)
                    
                    logger.info(f"当前学习到的mask长度: {mask_length:.6f} (factor: {factor_value:.6f}, 变化: {delta:.8f})")
                
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)

            # 将target_masks作为feature_masks传递给模型
            predictions = self.model(X.to(self.device), padding_masks, target_masks)  # (batch_size, padded_length, feat_dim)

            # 将padding_masks传递给损失函数，以便DenoisingImputationLoss可以正确处理
            loss = self.loss_module(predictions, targets, target_masks, padding_masks)
            
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch

            if keep_all:
                per_batch['targets'].append(targets.cpu().numpy())
                per_batch['predictions'].append(predictions.cpu().numpy())
                per_batch['metrics'].append([loss.cpu().numpy()])
                per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class SupervisedRunner(BaseRunner):

    def __init__(self, *args, **kwargs):
        super(SupervisedRunner, self).__init__(*args, **kwargs)
        self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.is_dual_branch = isinstance(self.model, DualTSTransformerEncoderClassifier)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):
            for j, e in enumerate(batch):
                if isinstance(e, Tensor):
                    batch[j] = e.to(self.device)
            if self.is_dual_branch:
                X1, X2, padding_masks1, padding_masks2, targets, IDs = batch  # 0s: ignore
                # classification: (batch_size, num_classes) of logits
                predictions = self.model(X1, padding_masks1, X2, padding_masks2)
            else:
                X, targets, padding_masks, IDs = batch  # 0s: ignore
                # 添加feature_masks=None参数保持接口一致
                predictions = self.model(X, padding_masks, feature_masks=None)  # for CNN1D_Classifier, padding_masks is not used

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # 添加mask_length正则化，鼓励mask_length_factor探索更广的值域
            if hasattr(self.model, 'mask_length_factor'):
                mask_reg_loss = mask_length_regularization_loss(self.model, center_value=0.5, strength=0.005)
                total_loss = total_loss + mask_reg_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # 使用梯度裁剪提高训练稳定性
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        
        # 记录当前学习到的信号增强值
        if hasattr(self.model, 'missing_signal_strength'):
            signal_value = self.model.missing_signal_strength.item()
            self.epoch_metrics['signal_strength'] = signal_value
            logger.info(f"当前学习到的信号增强值: {signal_value:.6f}")
            
            # 记录当前学习到的mask长度
            if hasattr(self.model, 'mask_length_factor'):
                factor_value = self.model.mask_length_factor.item()
                mask_length = 1 + 9 * torch.sigmoid(torch.tensor(factor_value)).item()  # 更新为1-10范围
                self.epoch_metrics['mask_length'] = mask_length
                self.epoch_metrics['mask_length_factor'] = factor_value
                # 计算上一个epoch的值（如果存在）
                prev_factor = getattr(self, 'prev_mask_factor', factor_value)
                delta = factor_value - prev_factor
                setattr(self, 'prev_mask_factor', factor_value)
                
                logger.info(f"当前学习到的mask长度: {mask_length:.6f} (factor: {factor_value:.6f}, 变化: {delta:.8f})")
                
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True, return_dfs=False):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        eval_start_time = time.time()
        for i, batch in enumerate(self.dataloader):
            for j, e in enumerate(batch):
                if isinstance(e, Tensor):
                    batch[j] = e.to(self.device)
            if self.is_dual_branch:
                X1, X2, padding_masks1, padding_masks2, targets, IDs = batch  # 0s: ignore
                # classification: (batch_size, num_classes) of logits
                predictions = self.model(X1, padding_masks1, X2, padding_masks2)
            else:
                X, targets, padding_masks, IDs = batch  # 0s: ignore
                # 添加feature_masks=None参数保持接口一致
                predictions = self.model(X.to(self.device), padding_masks, feature_masks=None)

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            per_batch['predictions'].append(predictions.cpu().detach().numpy())
            per_batch['metrics'].append([loss.cpu().detach().numpy()])
            per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch
        logger.info(f'eval time: {time.time() - eval_start_time} s')
        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(
            predictions)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()

        class_names = self.exp_config['class_names']

        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names, return_dfs=return_dfs)

        if return_dfs:
            return metrics_dict

        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes
        self.epoch_metrics['recall'] = metrics_dict['rec_avg']  # average recall over all classes
        self.epoch_metrics['f1'] = np.mean(metrics_dict['f1'])  # average f1 over all classes

        if self.model.num_classes == 2:
            false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
            self.epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

            prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
            self.epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics
