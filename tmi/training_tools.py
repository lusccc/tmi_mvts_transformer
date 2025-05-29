import numpy as np
import torch
from logzero import logger

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, monitor_on='loss'):
        """
        初始化早停模块
        
        Args:
            patience: 触发早停前等待的轮数
            verbose: 是否打印详细信息
            delta: 判断改进的最小变化
            monitor_on: 监控的指标，可以是'loss'、'accuracy'或'combined'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf
        self.delta = delta
        self.monitor_on = monitor_on

    def __call__(self, val_metrics):
        """
        根据给定的验证指标判断是否应该早停
        
        Args:
            val_metrics: 单个指标值或包含'loss'和'accuracy'的字典
            
        Returns:
            self: 更新后的EarlyStopping实例
        """
        # 处理不同的监控方式
        if self.monitor_on == 'combined':
            # 确保传入的是字典且包含所需的键
            if not isinstance(val_metrics, dict) or 'loss' not in val_metrics or 'accuracy' not in val_metrics:
                raise ValueError("当monitor_on='combined'时，val_metrics必须是包含'loss'和'accuracy'的字典")
            
            # 综合指标: -accuracy + loss (accuracy越高越好，loss越低越好)
            val_metric = val_metrics['loss'] - val_metrics['accuracy']
            score = -val_metric  # 转换为越大越好
        else:
            # 单一指标模式
            if isinstance(val_metrics, dict):
                val_metric = val_metrics.get(self.monitor_on)
                if val_metric is None:
                    raise ValueError(f"val_metrics中找不到键'{self.monitor_on}'")
            else:
                val_metric = val_metrics
            
            # 对于accuracy，转为负值使其与loss保持一致(越小越好)
            val_metric = -val_metric if self.monitor_on == 'accuracy' else val_metric
            score = -val_metric  # 转换为越大越好
            
        # 早停逻辑
        if self.best_score is None:
            # 第一次调用
            self.best_score = score
        elif score < self.best_score + self.delta:
            # 没有足够的改进
            self.counter += 1
            if self.verbose:
                logger.warn(f'EarlyStopping计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 有足够的改进
            self.best_score = score
            self.counter = 0
            if self.verbose:
                logger.info(f'EarlyStopping: 发现更好的性能，重置计数器。新最佳分数: {-score if self.monitor_on == "accuracy" else score}')
                
        return self

