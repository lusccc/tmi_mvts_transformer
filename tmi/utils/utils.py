import builtins
import functools
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import xlrd
import xlwt
from xlutils.copy import copy
import ipdb
import torch.nn as nn

from logzero import logger


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time} secs")
        return value

    return wrapper_timer


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def save_dual_branch_model(path, epoch, model, optimizer=None):
    """专门用于保存双分支Transformer模型的函数
    
    Args:
        path: 保存路径
        epoch: 当前训练轮次
        model: DualTSTransformerEncoderClassifier模型实例
        optimizer: 优化器实例（可选）
    """
    # 确保模型在同一设备上
    model_device = next(model.parameters()).device
    
    # 确保模型结构正确
    ensure_dual_branch_model_structure(model)
    
    # 创建保存字典
    data = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'model_hyperparams': {
            'num_classes': model.num_classes,
            'dropout': model.dropout1.p,
            'activation': model.act.__name__ if hasattr(model.act, '__name__') else str(model.act)
        },
        'trajectory_branch_hyperparams': {
            'd_model': model.trajectory_branch.d_model,
            'max_len': model.trajectory_branch.max_len,
            'feat_dim': model.trajectory_branch.feat_dim,
            'n_heads': model.trajectory_branch.n_heads,
            'dropout': model.trajectory_branch.dropout
        },
        'feature_branch_hyperparams': {
            'd_model': model.feature_branch.d_model,
            'max_len': model.feature_branch.max_len,
            'feat_dim': model.feature_branch.feat_dim,
            'n_heads': model.feature_branch.n_heads,
            'dropout': model.feature_branch.dropout
        }
    }
    
    # 验证分支的output_layer是否为Flatten类型
    from tmi.models.models import TSTransformerEncoderForDualBranch
    if not isinstance(model.trajectory_branch, TSTransformerEncoderForDualBranch):
        logger.warning('轨迹分支不是TSTransformerEncoderForDualBranch实例，可能导致加载问题')
    
    if not isinstance(model.feature_branch, TSTransformerEncoderForDualBranch):
        logger.warning('特征分支不是TSTransformerEncoderForDualBranch实例，可能导致加载问题')
    
    # 添加优化器状态（如果提供）
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    
    # 保存模型
    torch.save(data, path)
    logger.info(f'双分支模型已保存到: {path}')


def load_dual_branch_model(model, model_path, optimizer=None, resume=False):
    """专门用于加载双分支Transformer模型的函数
    
    Args:
        model: 目标DualTSTransformerEncoderClassifier模型实例
        model_path: 模型文件路径
        optimizer: 优化器实例（可选）
        resume: 是否恢复训练（如果为True，则加载优化器状态）
        
    Returns:
        tuple: (加载后的模型, 优化器, 开始epoch)
    """
    logger.info(f'加载双分支模型: {model_path}')
    start_epoch = 0
    
    # 加载模型文件
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    # 验证模型结构
    from tmi.models.models import DualTSTransformerEncoderClassifier, TSTransformerEncoderForDualBranch
    if not isinstance(model, DualTSTransformerEncoderClassifier):
        logger.error("提供的模型不是DualTSTransformerEncoderClassifier实例")
        raise TypeError("模型类型不匹配")
    
    # # 验证分支类型
    if not isinstance(model.trajectory_branch, TSTransformerEncoderForDualBranch):
        logger.warning("轨迹分支不是TSTransformerEncoderForDualBranch实例，性能可能受到影响")
        
    if not isinstance(model.feature_branch, TSTransformerEncoderForDualBranch):
        logger.warning("特征分支不是TSTransformerEncoderForDualBranch实例，性能可能受到影响")
    
    # 加载状态字典
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(f'已从{model_path}加载模型。训练轮次: {checkpoint["epoch"]}')
    
    # 检查模型结构
    ensure_dual_branch_model_structure(model)
    
    # 恢复优化器参数（如果需要）
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            logger.info(f'已恢复优化器状态，从轮次{start_epoch}继续训练')
        else:
            logger.warning('检查点中没有优化器参数')
    
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def ensure_dual_branch_model_structure(model):
    """检查DualTSTransformerEncoderClassifier模型结构
    
    确保两个分支都存在并且结构正确
    
    Args:
        model: DualTSTransformerEncoderClassifier模型实例
    """
    from tmi.models.models import DualTSTransformerEncoderClassifier, TSTransformerEncoderForDualBranch
    if not isinstance(model, DualTSTransformerEncoderClassifier):
        logger.warning("模型不是DualTSTransformerEncoderClassifier实例，跳过结构检查")
        return
    
    # 检查trajectory_branch
    if not hasattr(model, 'trajectory_branch'):
        logger.error("模型缺少trajectory_branch属性")
    else:
        # 检查是否为TSTransformerEncoderForDualBranch实例
        if not isinstance(model.trajectory_branch, TSTransformerEncoderForDualBranch):
            logger.warning("轨迹分支不是TSTransformerEncoderForDualBranch实例，性能可能受到影响")
            
        # 检查output_layer是否为Flatten类型
        if not hasattr(model.trajectory_branch, 'output_layer') or not isinstance(model.trajectory_branch.output_layer, nn.Flatten):
            logger.warning("轨迹分支的output_layer不是Flatten类型，可能导致处理问题")
    
    # 检查feature_branch
    if not hasattr(model, 'feature_branch'):
        logger.error("模型缺少feature_branch属性")
    else:
        # 检查是否为TSTransformerEncoderForDualBranch实例
        if not isinstance(model.feature_branch, TSTransformerEncoderForDualBranch):
            logger.warning("特征分支不是TSTransformerEncoderForDualBranch实例，性能可能受到影响")
            
        # 检查output_layer是否为Flatten类型
        if not hasattr(model.feature_branch, 'output_layer') or not isinstance(model.feature_branch.output_layer, nn.Flatten):
            logger.warning("特征分支的output_layer不是Flatten类型，可能导致处理问题")
    
    logger.info("双分支模型结构检查完成")


def load_model(model, model_path, optimizer=None, resume=False, change_output=False,
               lr=None, lr_step=None, lr_factor=None):
    logger.info(f'load model: {model_path}, change output: {change_output}')
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    if change_output:
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('output_layer'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

    return config


def save_model_hyperparams(config, hyperparams):
    hyperparams.pop('__class__')
    hyperparams.pop('self')
    output_dir = config['output_dir']
    with open(os.path.join(output_dir, f"{config['task']}_model_hyperparams.json"), 'w') as fp:
        json.dump(hyperparams, fp, indent=4, sort_keys=True)

    # try:
    #     with open(os.path.join('experiments', 'tmp', config['data_class'] + '_model_hyperparams.json'), 'w') as fp:
    #         json.dump(hyperparams, fp, indent=4, sort_keys=True)
    # except:
    #     logger.error('no tmp dir!')


def load_model_hyperparams(file_path_or_str):
    """加载模型超参数
    
    Args:
        file_path_or_str: 可以是JSON文件路径，也可以是字符串格式的超参数 
                         (例如 "feat_dim=4;max_len=128;d_model=128;n_heads=16")
    
    Returns:
        dict: 超参数字典
    """
    # 检查是否为字符串格式的超参数
    if isinstance(file_path_or_str, str) and ';' in file_path_or_str and '=' in file_path_or_str:
        try:
            # 解析格式为 "key1=value1;key2=value2;..." 的字符串
            hyperparams = {}
            for param in file_path_or_str.split(';'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 尝试将值转换为适当的数据类型
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            # 保持为字符串
                            pass
                            
                    hyperparams[key] = value
            
            logger.info(f"从字符串加载超参数: {hyperparams}")
            return hyperparams
        except Exception as e:
            logger.error(f"解析超参数字符串失败: {e}")
            raise ValueError(f"无法解析超参数字符串: {file_path_or_str}")
    
    # 如果不是字符串格式，则视为文件路径
    try:
        with open(file_path_or_str) as hp:
            hyperparams = json.load(hp)
        return hyperparams
    except Exception as e:
        logger.error(f"从文件加载超参数失败: {e}")
        raise ValueError(f"无法从文件加载超参数: {file_path_or_str}")


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def save_classification_results_to_excel(results_dict, output_dir):
    """
    将分类结果的三个DataFrame保存到Excel文件中的不同sheet。
    
    参数:
        results_dict: 包含三个DataFrame的字典:
            - cm_df: 混淆矩阵DataFrame
            - classification_report_df: 分类报告DataFrame 
            - prec_rec_histogram_df: 精确率-召回率直方图DataFrame
        output_dir: 输出目录路径
    """

    output_path = os.path.join(output_dir, 'classification_results.xlsx')

    # 使用ExcelWriter写入多个sheet
    with pd.ExcelWriter(output_path) as writer:
        results_dict['cm_df'].to_excel(writer, sheet_name='Confusion Matrix')
        results_dict['classification_report_df'].to_excel(writer, sheet_name='Classification Report')
        results_dict['prec_rec_histogram_df'].to_excel(writer, sheet_name='Precision Recall Histogram')

    logger.info(f"分类结果已保存到: {output_path}")


def export_performance_metrics(filepath, metrics_table, header, book=None, sheet_name="metrics"):
    """使用pandas DataFrame导出性能指标到Excel文件
    
    Args:
        filepath: 输出文件路径
        metrics_table: 指标数据列表
        header: 列名列表
        book: 忽略此参数(为了保持向后兼容)
        sheet_name: Excel的sheet名称
    """
    # 将数据转换为DataFrame
    df = pd.DataFrame(metrics_table, columns=header)
    
    # 导出到Excel
    df.to_excel(filepath, sheet_name=sheet_name, index=False)
    
    logger.info(f"已导出每轮性能指标到 '{filepath}'")
    
    return None  # 不再返回workbook对象


def write_row(sheet, row_ind, data_list):
    """Write a list to row_ind row of an excel sheet"""

    row = sheet.row(row_ind)
    for col_ind, col_value in enumerate(data_list):
        row.write(col_ind, col_value)
    return


def write_table_to_sheet(table, work_book, sheet_name=None):
    """Writes a table implemented as a list of lists to an excel sheet in the given work book object"""

    sheet = work_book.add_sheet(sheet_name)

    for row_ind, row_list in enumerate(table):
        write_row(sheet, row_ind, row_list)

    return work_book


def export_record(filepath, values):
    """Adds a list of values as a bottom row of a table in a given excel file"""

    read_book = xlrd.open_workbook(filepath, formatting_info=True)
    read_sheet = read_book.sheet_by_index(0)
    last_row = read_sheet.nrows

    work_book = copy(read_book)
    sheet = work_book.get_sheet(0)
    write_row(sheet, last_row, values)
    work_book.save(filepath)


def register_record(config, filepath, timestamp, experiment_name, best_metrics, final_metrics=None):
    """
    将实验的最佳和最终指标记录到Excel表格中。如果文件不存在则创建新文件。
    
    Args:
        filepath: 记录文件路径
        timestamp: 时间戳字符串
        experiment_name: 实验名称
        best_metrics: 最佳epoch的指标字典 {metric_name: metric_value}
        final_metrics: 最终epoch的指标字典 {metric_name: metric_value}
    """
    # 准备行数据
    metrics_names, metrics_values = zip(*best_metrics.items())
    row_data = {
        'Timestamp': timestamp,
        'Name': experiment_name
    }
    
    # 添加最佳指标
    for name, value in zip(metrics_names, metrics_values):
        row_data[f'Best {name}'] = value
        
    # 添加最终指标
    if final_metrics is not None:
        for name, value in final_metrics.items():
            row_data[f'Final {name}'] = value
            
    # 添加配置信息
    row_data.update({
        'DataClass': config['data_class'],
        'Task': config['task'],
        'DataInputType': config['input_type'],
        'TestOnly': config['test_only'],
        'ResultsPath': config['output_dir']
    })
    
    try:
        if os.path.exists(filepath):
            # 如果文件存在,读取并追加新行
            df = pd.read_excel(filepath)
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        else:
            # 如果文件不存在,创建新DataFrame
            logger.warning(f"记录文件 '{filepath}' 不存在! 创建新文件...")
            directory = os.path.dirname(filepath)
            if len(directory) and not os.path.exists(directory):
                os.makedirs(directory)
            df = pd.DataFrame([row_data])
            
        # 保存到Excel
        df.to_excel(filepath, index=False)
        logger.info(f"已导出性能记录到 '{filepath}'")
        
    except Exception as e:
        # 如果保存失败,尝试使用备用路径
        alt_path = os.path.join(os.path.dirname(filepath), f"record_{experiment_name}.xlsx")
        logger.error(f"保存到 '{filepath}' 失败! 改为保存到: {alt_path}")
        df.to_excel(alt_path, index=False)


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


# def check_model1(model, verbose=False, stop_on_error=False):
#     status_ok = True
#     for name, param in model.named_parameters():
#         nan_grads = torch.isnan(param.grad)
#         nan_params = torch.isnan(param)
#         if nan_grads.any() or nan_params.any():
#             status_ok = False
#             print("Param {}: {}/{} nan".format(name, torch.sum(nan_params), param.numel()))
#             if verbose:
#                 print(param)
#             print("Grad {}: {}/{} nan".format(name, torch.sum(nan_grads), param.grad.numel()))
#             if verbose:
#                 print(param.grad)
#             if stop_on_error:
#                 ipdb.set_trace()
#     if status_ok:
#         print("Model Check: OK")
#     else:
#         print("Model Check: PROBLEM")


def check_model(model, verbose=False, zero_thresh=1e-8, inf_thresh=1e6, stop_on_error=False):
    status_ok = True
    for name, param in model.named_parameters():
        param_ok = check_tensor(param, verbose=verbose, zero_thresh=zero_thresh, inf_thresh=inf_thresh)
        if not param_ok:
            status_ok = False
            print("Parameter '{}' PROBLEM".format(name))
        grad_ok = True
        if param.grad is not None:
            grad_ok = check_tensor(param.grad, verbose=verbose, zero_thresh=zero_thresh, inf_thresh=inf_thresh)
        if not grad_ok:
            status_ok = False
            print("Gradient of parameter '{}' PROBLEM".format(name))
        if stop_on_error and not (param_ok and grad_ok):
            ipdb.set_trace()

    if status_ok:
        print("Model Check: OK")
    else:
        print("Model Check: PROBLEM")


def check_tensor(X, verbose=True, zero_thresh=1e-8, inf_thresh=1e6):
    is_nan = torch.isnan(X)
    if is_nan.any():
        print("{}/{} nan".format(torch.sum(is_nan), X.numel()))
        return False

    num_small = torch.sum(torch.abs(X) < zero_thresh)
    num_large = torch.sum(torch.abs(X) > inf_thresh)

    if verbose:
        print("Shape: {}, {} elements".format(X.shape, X.numel()))
        print("No 'nan' values")
        print("Min: {}".format(torch.min(X)))
        print("Median: {}".format(torch.median(X)))
        print("Max: {}".format(torch.max(X)))

        print("Histogram of values:")
        values = X.view(-1).detach().numpy()
        hist, binedges = np.histogram(values, bins=20)
        for b in range(len(binedges) - 1):
            print("[{}, {}): {}".format(binedges[b], binedges[b + 1], hist[b]))

        print("{}/{} abs. values < {}".format(num_small, X.numel(), zero_thresh))
        print("{}/{} abs. values > {}".format(num_large, X.numel(), inf_thresh))

    if num_large:
        print("{}/{} abs. values > {}".format(num_large, X.numel(), inf_thresh))
        return False

    return True


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def recursively_hook(model, hook_fn):
    for name, module in model.named_children():  # model._modules.items():
        if len(list(module.children())) > 0:  # if not leaf node
            for submodule in module.children():
                recursively_hook(submodule, hook_fn)
        else:
            module.register_forward_hook(hook_fn)


def compute_loss(net: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 device: torch.device = 'cpu') -> torch.Tensor:
    """Compute the loss of a network on a given dataset.

    Does not compute gradient.

    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.

    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            netout = net(x.to(device)).cpu()
            running_loss += loss_function(y, netout)

    return running_loss / len(dataloader)
