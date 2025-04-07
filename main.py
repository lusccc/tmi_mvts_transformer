import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
# to address Too many open files Pin memory thread exited unexpectedly:
import torch.multiprocessing
from logzero import logger
from torch.backends import cudnn
from torch.profiler import tensorboard_trace_handler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 3rd party packages
from tqdm import tqdm
from imblearn.combine import SMOTEENN

from tmi.ml_classification import MLClassifier, calc_handcrafted_features

# Project modules
from tmi.datasets import dataset
from tmi.datasets.data import data_factory, Normalizer
from tmi.datasets.datasplit import split_dataset
from tmi.models.loss import get_loss_module
from tmi.models.models import model_factory
from tmi.optimizers import get_optimizer
from tmi.options import Options
from tmi.runner import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from tmi.training_tools import EarlyStopping
from tmi.utils import utils


class TrainingPipeline:
    def __init__(self, config):
        """初始化训练流程"""
        self.config = config
        self.total_epoch_time = 0
        self.total_eval_time = 0
        self.total_start_time = time.time()
        
        # 记录运行命令
        logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))
        
        # 设置随机种子
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
            
        # 设置设备
        self._setup_device()
        
        # 初始化必要的变量
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.model = None
        self.optimizer = None
        self.loss_module = None
        self.tensorboard_writer = None

    def _setup_device(self):
        """设置计算设备"""
        device = 'cuda' if (torch.cuda.is_available() and self.config['gpu'] != '-1') else 'cpu'
        if device != 'cuda':
            logger.error("no cuda!! ")
            device = torch.device(device)
        else:
            device = torch.device(device)
            logger.info("Using device: {}".format(device))
            logger.info("Device index: {}".format(torch.cuda.current_device()))
        self.device = device

    def setup_data(self):
        """加载和预处理数据"""
        logger.info("Loading and preprocessing data ...")
        
        # 1. 加载数据
        data_class = data_factory[self.config['data_class']]
        self.train_data = data_class(limit_size=self.config['limit_size'], config=self.config, data_split='train')
        self.test_data = data_class(limit_size=self.config['limit_size'], config=self.config, data_split='test')
        
        # 确定验证方法
        if 'classification' in self.config['task']:
            validation_method = 'ShuffleSplit'
            labels = self.train_data.labels_df.values.flatten()
        else:
            validation_method = 'ShuffleSplit'
            labels = None

        # 2. 分割数据集
        self.test_indices = self.test_data.all_IDs
        self.val_data = self.train_data  # 会被val_indices过滤
        self.val_indices = []

        if self.config['val_ratio'] > 0:
            self.train_indices, self.val_indices, _ = split_dataset(
                data_indices=self.train_data.all_IDs,
                validation_method=validation_method,
                n_splits=1,
                validation_ratio=self.config['val_ratio'],
                test_set_ratio=None,
                test_indices=self.test_indices,
                random_seed=10086,
                labels=labels
            )
            self.train_indices = self.train_indices[0]
            self.val_indices = self.val_indices[0]
        else:
            self.train_indices = self.train_data.all_IDs
            if self.test_indices is None:
                self.test_indices = []

        # 3. 输出数据集大小信息
        logger.info("{} samples may be used for training".format(len(self.train_indices)))
        logger.info("{} samples will be used for validation".format(len(self.val_indices)))
        logger.info("{} samples will be used for testing".format(len(self.test_indices)))

        # 4. 保存数据索引
        self._save_data_indices()
        
        # 5. 对特征进行预处理
        self._preprocess_features()
    
    def _save_data_indices(self):
        """保存数据分割索引"""
        with open(os.path.join(self.config['output_dir'], 'data_indices.json'), 'w') as f:
            try:
                json.dump({
                    'train_indices': list(map(int, self.train_indices)),
                    'val_indices': list(map(int, self.val_indices)),
                    'test_indices': list(map(int, self.test_indices))
                }, f, indent=4)
            except ValueError:  # 处理非整数索引
                json.dump({
                    'train_indices': list(self.train_indices),
                    'val_indices': list(self.val_indices),
                    'test_indices': list(self.test_indices)
                }, f, indent=4)
    
    def _preprocess_features(self):
        """预处理特征"""
        for (train_df, train_normalization), (test_df, test_normalization) in zip(
                self.train_data.feature_dfs, self.test_data.feature_dfs):
            normalizer = Normalizer(train_normalization)
            train_df.loc[self.train_indices] = normalizer.normalize(train_df.loc[self.train_indices])
            if len(self.val_indices):
                train_df.loc[self.val_indices] = normalizer.normalize(train_df.loc[self.val_indices])
            test_df.loc[self.test_indices] = normalizer.normalize(test_df.loc[self.test_indices])

    def setup_dl_model(self):
        """创建和初始化深度学习模型"""
        logger.info("Creating model ...")
        self.model = model_factory(self.config, self.train_data)

        # 设置冻结层
        if self.config['freeze']:
            for name, param in self.model.named_parameters():
                if 'output_layer' in name:
                    logger.info(f'set layer {name} requires_grad = True')
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # 输出模型信息
        logger.info("Model:\n{}".format(self.model))
        logger.info("Total number of parameters: {}".format(utils.count_parameters(self.model)))
        logger.info("Trainable parameters: {}".format(utils.count_parameters(self.model, trainable=True)))

        # 初始化优化器
        self._setup_optimizer()
        
        # 加载模型状态
        self._load_model_state()
        
        # 设置损失函数
        self.loss_module = get_loss_module(self.config)
        
    def _setup_optimizer(self):
        """设置优化器"""
        if self.config['global_reg']:
            weight_decay = self.config['l2_reg']
            self.output_reg = None
        else:
            weight_decay = 0
            self.output_reg = self.config['l2_reg']

        optim_class = get_optimizer(self.config['optimizer'])
        self.optimizer = optim_class(
            self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=weight_decay
        )
        
        # 学习率相关参数
        self.start_epoch = 0
        self.lr_step = 0  # current step index of `lr_step`
        self.lr = self.config['lr']  # current learning step
        
    def _load_model_state(self):
        """加载模型状态"""
        # 加载多分支模型
        if self.config['task'] == 'dual_branch_classification':
            if self.config['load_trajectory_branch']:
                self.model.trajectory_branch, _, __ = utils.load_model(
                    self.model.trajectory_branch,
                    self.config['load_trajectory_branch'],
                    self.optimizer,
                    self.config['resume'],
                    self.config['change_output'],
                    self.config['lr'],
                    self.config['lr_step'],
                    self.config['lr_factor']
                )
            if self.config['load_feature_branch']:
                self.model.feature_branch, _, __ = utils.load_model(
                    self.model.feature_branch,
                    self.config['load_feature_branch'],
                    self.optimizer,
                    self.config['resume'],
                    self.config['change_output'],
                    self.config['lr'],
                    self.config['lr_step'],
                    self.config['lr_factor']
                )

        # 加载完整模型
        if self.config['load_model']:
            self.model, self.optimizer, self.start_epoch = utils.load_model(
                self.model,
                self.config['load_model'],
                self.optimizer,
                self.config['resume'],
                self.config['change_output'],
                self.config['lr'],
                self.config['lr_step'],
                self.config['lr_factor']
            )
        
        # 将模型移至目标设备
        self.model.to(self.device)
        
    def prepare_data_loaders(self):
        """准备数据加载器，区分ML和DL模型的需求"""
        # 设置数据加载器
        self.dataset_class, self.collate_fn, self.runner_class = pipeline_factory(self.config)
        
        # 测试集数据加载器
        if self.config['test_only'] == 'testset' or self.config['task'] == 'ml_classification':
            test_dataset = self.dataset_class(self.test_data, self.test_indices)
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True,
                collate_fn=lambda x: self.collate_fn(x, )
            )
            
        # 验证集数据加载器
        val_dataset = self.dataset_class(self.val_data, self.val_indices)
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        # 训练集数据加载器
        train_dataset = self.dataset_class(self.train_data, self.train_indices)
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        # 加速数据加载
        # logger.info('data loader to list ...')
        # self.val_loader = list(self.val_loader)
        # self.train_loader = list(self.train_loader)
        
        # 保存数据集引用
        self.train_dataset = train_dataset
        

        
    def evaluate_dl_test(self):
        """评估深度学习模型在测试集上的性能"""
        test_evaluator = self.runner_class(
            self.model, 
            self.test_loader, 
            self.device, 
            self.loss_module,
            exp_config=self.config
        )
        aggr_metrics_test = test_evaluator.evaluate(keep_all=True, return_dfs=True)
        # 保存分类结果到Excel
        utils.save_classification_results_to_excel(aggr_metrics_test, self.config['output_dir'])
        
    def train_dl_model(self):
        """训练深度学习模型"""
        # 初始化训练器和评估器
        trainer = self.runner_class(
            self.model, 
            self.train_loader, 
            self.device, 
            self.loss_module, 
            self.optimizer, 
            l2_reg=self.output_reg,
            exp_config=self.config
        )
        
        val_evaluator = self.runner_class(
            self.model, 
            self.val_loader, 
            self.device, 
            self.loss_module,
            exp_config=self.config
        )
        
        # 初始化TensorBoard
        self.tensorboard_writer = SummaryWriter(self.config['tensorboard_dir'])
        
        # 初始化训练记录
        best_value = 1e16 if self.config['key_metric'] in NEG_METRICS else -1e16
        metrics = []  # 存储每个epoch的验证指标
        best_metrics = {}
        
        # 在训练前先评估验证集
        metrics_names = None
        if self.config['val_ratio'] > 0:
            aggr_metrics_val, best_metrics, best_value = validate(
                val_evaluator, 
                self.tensorboard_writer, 
                self.config, 
                best_metrics,
                best_value, 
                epoch=0
            )
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))
        else:
            if 'classification' in self.config['task']:
                metrics_names = ('epoch', 'loss', 'accuracy', 'precision')
            else:
                metrics_names = ('epoch', 'loss')
        
        # 设置早停
        monitor_on = 'accuracy' if 'classification' in self.config['task'] else 'loss'
        early_stopping = EarlyStopping(
            round(self.config['patience'] / self.config['val_interval']), 
            verbose=True,
            monitor_on=monitor_on
        )
        
        # 开始训练循环
        logger.info('Starting training...')
        early_stop = False
        
        for epoch in tqdm(range(self.start_epoch + 1, self.config["epochs"] + 1), desc='Training Epoch', leave=False):
            # 设置保存标记
            mark = epoch if self.config['save_all'] else 'last'
            
            # 训练一个epoch
            epoch_start_time = time.time()
            aggr_metrics_train = trainer.train_epoch(epoch)
            epoch_runtime = time.time() - epoch_start_time
            
            # 记录训练指标
            print_str = 'Epoch {} Training Summary: '.format(epoch)
            for k, v in aggr_metrics_train.items():
                self.tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
                print_str += '{}: {:8f} | '.format(k, v)
            logger.info(print_str)
            
            # 记录时间信息
            logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(
                *utils.readable_time(epoch_runtime))
            )
            self.total_epoch_time += epoch_runtime
            avg_epoch_time = self.total_epoch_time / (epoch - self.start_epoch)
            avg_batch_time = avg_epoch_time / len(self.train_loader)
            avg_sample_time = avg_epoch_time / len(self.train_dataset)
            
            logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(
                *utils.readable_time(avg_epoch_time))
            )
            logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
            logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))
            
            # 定期在验证集上评估
            if self.config['val_ratio'] > 0 and \
                    ((epoch == self.config["epochs"]) or 
                     (epoch == self.start_epoch + 1) or 
                     (epoch % self.config['val_interval'] == 0)):
                aggr_metrics_val, best_metrics, best_value = validate(
                    val_evaluator, 
                    self.tensorboard_writer, 
                    self.config,
                    best_metrics, 
                    best_value, 
                    epoch
                )
                metrics_names, metrics_values = zip(*aggr_metrics_val.items())
                metrics.append(list(metrics_values))
                
                # 检查早停条件
                if early_stopping(aggr_metrics_val[monitor_on]).early_stop:
                    early_stop = True
                    logger.warning('early stopping reached')
            
            # 保存模型
            utils.save_model(
                os.path.join(self.config['save_dir'], 'model_{}.pth'.format(mark)), 
                epoch, 
                self.model, 
                self.optimizer
            )
            
            if early_stop:
                logger.warning('stopping training...')
                break
                
            # 学习率调度
            self._adjust_learning_rate(epoch)
            
        return metrics, metrics_names, best_metrics, best_value, aggr_metrics_val
    
    def _adjust_learning_rate(self, epoch):
        """调整学习率"""
        if epoch == self.config['lr_step'][self.lr_step]:
            # 在学习率改变时保存模型
            utils.save_model(
                os.path.join(self.config['save_dir'], 'model_{}.pth'.format(epoch)), 
                epoch, 
                self.model,
                self.optimizer
            )
            
            # 更新学习率
            self.lr = self.lr * self.config['lr_factor'][self.lr_step]
            if self.lr_step < len(self.config['lr_step']) - 1:
                self.lr_step += 1
                
            logger.info('Learning rate updated to: {}'.format(self.lr))
            
            # 应用新的学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
                
    def export_dl_train_history(self, metrics, metrics_names, best_metrics,
                                 best_value, aggr_metrics_val):
        """导出训练结果"""
        # 导出性能指标
        metrics_filepath = os.path.join(
            self.config["output_dir"], 
            "metrics_history_" + self.config["experiment_name"] + ".xls"
        )
        book = utils.export_performance_metrics(
            metrics_filepath, 
            metrics, 
            metrics_names, 
            sheet_name="metrics"
        )
        
        # 记录实验结果
        utils.register_record(
            self.config, 
            self.config["records_file"], 
            self.config["initial_timestamp"], 
            self.config["experiment_name"],
            best_metrics, 
            aggr_metrics_val
        )
        
        # 记录最佳性能
        logger.info('Best {} was {}. Other metrics: {}'.format(
            self.config['key_metric'], 
            best_value, 
            str(best_metrics)
        ))
        
    def setup_ml_model(self):
        """设置机器学习模型"""
        logger.info("创建机器学习模型...")
        
        # 初始化MLClassifier
        self.ml_classifier = MLClassifier(self.config)
        
        logger.info("机器学习模型将在训练阶段通过网格搜索创建")
        
        # 创建模型保存目录
        self.ml_model_dir = os.path.join(self.config['save_dir'], 'ml_models')
        if not os.path.exists(self.ml_model_dir):
            os.makedirs(self.ml_model_dir)
            
        logger.info(f"机器学习模型将保存在: {self.ml_model_dir}")
    
    def _prepare_ml_data(self, data_loader):
        """辅助方法：从dataloader中准备机器学习所需的数据
        
        Args:
            data_loader: DataLoader实例
            
        Returns:
            tuple: (特征数据X, 标签数据y)
        """
        batches = [(batch[0].numpy(), batch[1].numpy()) for batch in data_loader]
        X = np.vstack([batch[0] for batch in batches])
        y = np.vstack([batch[1] for batch in batches])
        return X, y

    def train_ml_model(self):
        """训练机器学习模型并在验证集上评估"""
        logger.info("准备机器学习训练数据...")
        
        # 准备训练和验证数据
        X_train, y_train = self._prepare_ml_data(self.train_loader)
        X_val, y_val = self._prepare_ml_data(self.val_loader)
        
        # 计算手工特征
        logger.info("计算手工特征...")
        x_handcrafted_train = self.ml_classifier.calc_handcrafted_features(X_train)
        x_handcrafted_val = self.ml_classifier.calc_handcrafted_features(X_val)
        
        # 训练模型
        logger.info("开始训练机器学习模型...")
        best_models, best_scores = self.ml_classifier.train_ml_models(
            x_handcrafted_train, 
            x_handcrafted_val, 
            y_train, 
            y_val
        )
        
        logger.info("机器学习模型训练完成!")
        return best_models, best_scores
    
    def evaluate_ml_test(self):
        """评估机器学习模型在测试集上的性能"""
        logger.info("准备机器学习测试数据...")
        
        # 准备测试数据
        X_test, y_test = self._prepare_ml_data(self.test_loader)
        
        # 计算手工特征
        x_handcrafted_test = self.ml_classifier.calc_handcrafted_features(X_test)
        
        # 评估模型
        logger.info("在测试集上评估机器学习模型...")
        test_results = self.ml_classifier.evaluate_ml_models(
            x_handcrafted_test, 
            y_test
        )
        
        logger.info("机器学习模型评估完成!")
        return test_results

    def run(self):
        """运行整个训练流程，根据任务类型区分ML和DL"""
        # 1. 加载和预处理数据
        self.setup_data()
        
        # 2. 准备数据加载器
        self.prepare_data_loaders()
        
        # 3. 根据任务类型执行不同操作
        if self.config['task'] == 'ml_classification':
            # 机器学习分类任务流程
            logger.info('执行机器学习分类任务...')
            
            # 设置机器学习模型
            self.setup_ml_model()
            
            if self.config['test_only'] == 'testset':
                # 仅在测试集上评估机器学习模型
                test_results = self.evaluate_ml_test()
                logger.info('机器学习模型测试完成!')
                return
            
            # 训练机器学习模型
            best_models, best_scores = self.train_ml_model()
            
            # # 在测试集上评估
            # test_results = self.evaluate_ml_test()
            
            # 完成
            logger.info('机器学习分类完成!')
        else:
            # 深度学习模型流程
            logger.info('执行深度学习模型训练/测试...')
            
            # 创建深度学习模型
            self.setup_dl_model()
            
            if self.config['test_only'] == 'testset':
                # 仅在测试集上评估深度学习模型
                self.evaluate_dl_test()
                logger.info('深度学习模型测试完成!')
                return
            
            # 训练深度学习模型
            metrics, metrics_names, best_metrics, best_value, aggr_metrics_val = self.train_dl_model()
            
            # 导出结果
            self.export_dl_train_history(metrics, metrics_names, best_metrics, best_value, aggr_metrics_val)
            
            # 完成
            logger.info('深度学习模型训练完成!')
        
        # 记录总运行时间
        total_runtime = time.time() - self.total_start_time
        logger.info("总运行时间: {} 小时, {} 分钟, {} 秒\n".format(*utils.readable_time(total_runtime)))
        


def main(config):
    """主函数，初始化和运行训练流程"""
    pipeline = TrainingPipeline(config)
    return pipeline.run()


if __name__ == '__main__':
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    dataset.config = config
    cudnn.benchmark = True
    main(config)