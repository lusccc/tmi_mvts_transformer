import argparse
import os
import pathlib
import pickle

import logzero
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd
from tmi.utils.analysis import Analyzer


from logzero import logger


def calc_handcrafted_features(feature_segments):
    handcrafted_features_segments = []
    HC = 19  # Heading rate threshold
    VS = 3.4  # Stop rate threshold
    VR = 0.26  # VCR threshold
    for single_segment in feature_segments:
        '''
        new:
        0        1     2         3         4             5     6        7               8 
        delta_t, hour, distance, velocity, acceleration, jerk, heading, heading_change, heading_change_rate
        '''
        # old:
        # 0        1     2  3  4  5  6   7    8  9
        # delta_t, hour, d, v, a, h, hc, hcr, s, tn
        delta_ts = single_segment[:, 0]
        dists = single_segment[:, 2]
        vs = single_segment[:, 3]
        accs = single_segment[:, 4]
        hcs = single_segment[:, 6]
        hcrs = single_segment[:, 7]

        length = np.sum(dists)
        avg_v = np.sum(dists) / np.sum(delta_ts)
        exp_v = np.mean(vs)
        var_v = np.var(vs)

        sorted_vs = np.sort(vs)[::-1]  # descending order
        max_v1s = sorted_vs[0]
        max_v2s = sorted_vs[1]
        max_v3s = sorted_vs[2]

        sorted_accs = np.sort(accs)[::-1]  # descending order
        max_a1s = sorted_accs[0]
        max_a2s = sorted_accs[1]
        max_a3s = sorted_accs[2]

        sorted_hcrs = np.sort(hcrs)[::-1]  # descending order
        max_h1s = sorted_hcrs[0]
        max_h2s = sorted_hcrs[1]
        max_h3s = sorted_hcrs[2]

        avg_hcrs = np.sum(hcs) / np.sum(delta_ts)
        exp_hcrs = np.mean(hcrs)
        exp_hcrs = np.var(hcrs)

        # Heading change rate (HCR)
        Pc = sum(1 for item in list(hcrs) if item > HC)
        # Stop Rate (SR)
        Ps = sum(1 for item in list(vs) if item < VS)
        # Velocity Change Rate (VCR)
        Pv = sum(1 for item in list(accs) if item > VR)

        # length, avg_v, exp_v, var_v, max_v1s, max_v2s, max_v3s, max_a1s, max_a2s, max_a3s, max_h1s, max_h2s,
        #    max_h3s, avg_hcrs, exp_hcrs, var_hcrs
        handcrafted_features_segments.append(
            [length, avg_v, exp_v, var_v, max_v1s, max_v2s, max_v3s, max_a1s, max_a2s, max_a3s, Pc * 1. / length,
             Ps * 1. / length, Pv * 1. / length])
    return np.array(handcrafted_features_segments)


class MLClassifier:
    def __init__(self, config):
        """初始化MLClassifier类
        
        参数:
            config: 配置字典，包含class_names等参数
        """
        self.config = config
        self.class_names = self.config['class_names']
        self.analyzer = Analyzer(print_conf_mat=True)
        
        # 统一定义所有模型和参数网格
        self.models_params = {
            RandomForestClassifier(): {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            KNeighborsClassifier(): {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            MLPClassifier(): {
                'hidden_layer_sizes': [(100,), (50, 25)],
                'activation': ['relu'],
                'learning_rate': ['adaptive']
            },
            DecisionTreeClassifier(): {
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'criterion': ['gini']
            },
            SVC(): {
                'C': [1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf']
            }
        }
        
        # 设置模型目录
        if 'load_model' in self.config and self.config['load_model']:
            self.model_dir = self.config['load_model']
            logger.info(f"将从 {self.model_dir} 加载模型进行测试")
        else:
            self.model_dir = os.path.join(self.config['save_dir'], 'ml_models')
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            logger.info(f"模型将保存在 {self.model_dir}")
            
        self.output_dir = self.config['output_dir']

    def calc_handcrafted_features(self, feature_segments):
        """计算手工特征
        
        参数:
            feature_segments: 特征片段
            
        返回:
            numpy.ndarray: 计算出的手工特征
        """
        return calc_handcrafted_features(feature_segments)

    def train_ml_models(self, x_handcrafted_train, x_handcrafted_val, y_train, y_val):
        """训练所有机器学习模型"""
        # 合并训练集和验证集
        X = np.vstack((x_handcrafted_train, x_handcrafted_val))
        y = np.vstack((y_train, y_val)).ravel()
        
        best_models = {}
        best_scores = {}
        best_params = {}
        params_df = pd.DataFrame(columns=['Model', 'Parameter', 'Value'])
        
        # 统一训练所有模型
        for model, param_grid in self.models_params.items():
            # 获取模型名称
            model_name = model.__class__.__name__
            if isinstance(model, Pipeline):
                model_name = model.steps[-1][1].__class__.__name__
            logger.info(f'训练 {model_name} 模型...')
            
            # 使用GridSearchCV进行参数搜索
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            
            # 打印和保存最佳参数
            logger.info(f"Best parameters for {model_name}:")
            logger.info(grid_search.best_params_)
            
            # 收集最佳参数
            best_params[model_name] = grid_search.best_params_
            for param, value in grid_search.best_params_.items():
                # 移除pipeline前缀(如果存在)
                param = param.split('__')[-1]
                params_df = pd.concat([params_df, pd.DataFrame({
                    'Model': [model_name],
                    'Parameter': [param],
                    'Value': [value]
                })], ignore_index=True)
            
            # 保存最佳模型
            model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(grid_search.best_estimator_, f)
            
            # 在验证集上评估
            y_val_pred = grid_search.best_estimator_.predict(x_handcrafted_val)
            logger.info(f"\n在验证集上评估 {model_name} 模型:")
            val_results = self.analyzer.analyze_classification(
                y_pred=y_val_pred,
                y_true=np.squeeze(y_val),
                class_names=self.class_names,
                return_dfs=True
            )
            
            # 记录交叉验证和验证集的结果
            best_models[model_name] = grid_search.best_estimator_
            best_scores[model_name] = {
                'cv_results': grid_search.cv_results_,
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'validation_results': val_results  # 添加验证集结果
            }
        
        # 保存结果到Excel
        self._save_results_to_excel(params_df, best_scores)
        
        return best_models, best_scores

    def evaluate_ml_models(self, x_handcrafted_test, y_test):
        """在测试集上评估之前训练好的机器学习模型
        
        参数:
            x_handcrafted_test: 测试集特征
            y_test: 测试集标签
            
        返回:
            test_results: 字典，包含每个模型在测试集上的性能指标
        """
        
        # 要评估的模型列表
        model_names = ['RandomForestClassifier', 'KNeighborsClassifier', 'MLPClassifier', 
                      'DecisionTreeClassifier', 'SVC']
        
        test_results = {}
        
        for model_name in model_names:
            # 加载模型
            model_path = os.path.join(self.model_dir, f'{model_name}.pkl')
            if not os.path.exists(model_path):
                logger.warning(f"模型文件不存在: {model_path}")
                continue
                
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            # 在测试集上预测
            y_pred = model.predict(x_handcrafted_test)
            
            # 使用 Analyzer 进行评估
            logger.info(f"\n###########{model_name} 评估结果:")
            results = self.analyzer.analyze_classification(
                y_pred=y_pred,
                y_true=np.squeeze(y_test),
                class_names=self.class_names,
                return_dfs=True
            )
            
            # 保存结果
            test_results[model_name] = results
        
        # 将所有结果保存到一个Excel文件中
        excel_path = os.path.join(self.output_dir, 'ml_evaluation_results.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            # 保存所有模型的测试结果
            all_results_df = pd.DataFrame()
            for model_name, result in test_results.items():
                model_df = result['classification_report_df'].copy()
                model_df.insert(0, 'Model', model_name)
                all_results_df = pd.concat([all_results_df, model_df], ignore_index=True)
            all_results_df.to_excel(writer, sheet_name='Test Results', index=False)
            
            # 保存每个模型的混淆矩阵
            for model_name, result in test_results.items():
                sheet_name = f'{model_name}_Confusion_Matrix'
                result['cm_df'].to_excel(writer, sheet_name=sheet_name)
        
        return test_results

    def _save_results_to_excel(self, params_df, best_scores):
        # 保存最佳参数
        excel_path = os.path.join(self.output_dir, 'ml_training_results.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            params_df.to_excel(writer, sheet_name='Best Parameters', index=False)
            
            # 保存验证结果
            all_val_results_df = pd.DataFrame()
            for model_name, result in best_scores.items():
                # 将字典转换为DataFrame
                model_df = pd.DataFrame(result['cv_results'])
                model_df.insert(0, 'Model', model_name)
                all_val_results_df = pd.concat([all_val_results_df, model_df], ignore_index=True)
            all_val_results_df.to_excel(writer, sheet_name='Validation Results', index=False)

