import torch
import torch.nn as nn
from torch.nn import functional as F
import logzero
from logzero import logger

def get_loss_module(config):
    task = config['task']
    disable_mask = config.get('disable_mask', False)

    if task == 'denoising_imputation_pretrain':
        # 对联合任务使用新的损失函数
        imputation_weight = config.get('imputation_weight', 1.0)
        denoising_weight = config.get('denoising_weight', 1.0)
        return DenoisingImputationLoss(
            imputation_weight=imputation_weight, 
            denoising_weight=denoising_weight, 
            reduction='none',
            disable_mask=disable_mask
        )
    elif task in ['imputation_pretrain', 'denoising_pretrain']:
        return MaskedMSELoss(reduction='none', disable_mask=disable_mask)  # outputs loss for each batch element
    elif task == 'trajrl_pretrain':
        # TrajRL预训练使用GPS坐标重建损失
        return TrajRLPretrainLoss(reduction='none', disable_mask=disable_mask)

    if "classification" in task:
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean', disable_mask: bool = False):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)
        self.disable_mask = disable_mask
        logger.info(f"MaskedMSELoss disable_mask: {self.disable_mask}")

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """
        
        # 如果disable_mask为True，则直接计算MSE而不使用mask
        if self.disable_mask:
            return self.mse_loss(y_pred, y_true)
              
        # 正常的masked loss计算
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


class DenoisingImputationLoss(nn.Module):
    """ 
    组合去噪和插补的损失函数
    - 对插补任务：只计算缺失位置(target_masks=1)的重建损失
    - 对去噪任务：计算非缺失位置(target_masks=0且在padding范围内)的重建损失
    """

    def __init__(self, imputation_weight: float = 1.0, denoising_weight: float = 1.0, 
                 reduction: str = 'mean', disable_mask: bool = False):
        super().__init__()

        self.imputation_weight = imputation_weight
        self.denoising_weight = denoising_weight
        self.reduction = reduction
        self.disable_mask = disable_mask
        self.mse_loss = nn.MSELoss(reduction=self.reduction)
        
        logger.info(f"DenoisingImputationLoss initialized with:")
        logger.info(f"  - imputation_weight: {self.imputation_weight}")
        logger.info(f"  - denoising_weight: {self.denoising_weight}")
        logger.info(f"  - disable_mask: {self.disable_mask}")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                target_masks: torch.BoolTensor, padding_masks: torch.BoolTensor = None) -> torch.Tensor:
        """
        计算联合的去噪和插补损失
        
        Args:
            y_pred: 模型预测值 (batch_size, seq_len, feat_dim)
            y_true: 真实目标值 (batch_size, seq_len, feat_dim) 
            target_masks: 插补掩码 (batch_size, seq_len, feat_dim) 1表示需要预测的位置(即缺失位置)，0表示非缺失位置
            padding_masks: 填充掩码 (batch_size, seq_len) 1表示有效位置，0表示填充位置
            
        Returns:
            计算得到的损失值
        """
        
        # 如果disable_mask为True，则直接计算全部MSE
        if self.disable_mask:
            # 忽略padding位置，只计算有效位置的MSE
            if padding_masks is not None:
                valid_mask = padding_masks.unsqueeze(-1).expand_as(y_pred)
                valid_pred = torch.masked_select(y_pred, valid_mask)
                valid_true = torch.masked_select(y_true, valid_mask)
                return self.mse_loss(valid_pred, valid_true)
            else:
                return self.mse_loss(y_pred, y_true)
        
        # 准备padding掩码，如果未提供则假设全部有效
        if padding_masks is None:
            padding_masks = torch.ones(y_pred.shape[0], y_pred.shape[1], dtype=torch.bool, device=y_pred.device)
        
        # 扩展padding_masks使其维度与特征维度匹配
        valid_mask = padding_masks.unsqueeze(-1).expand_as(y_pred)
        
        # 当reduction为'none'时，我们需要保持与原始MaskedMSELoss一致的返回维度
        if self.reduction == 'none':
            # 创建一个与y_pred相同形状的空损失张量
            total_losses = torch.zeros_like(y_pred)
            
            # 计算插补损失 (针对target_masks=1的位置，即缺失位置)
            imputation_positions = target_masks & valid_mask  # 缺失且有效的位置
            # 只计算imputation位置的均方误差，其他位置保持为0
            if torch.any(imputation_positions):
                squared_diff = (y_pred - y_true) ** 2
                total_losses[imputation_positions] += self.imputation_weight * squared_diff[imputation_positions]
            
            # 计算去噪损失 (针对target_masks=0的有效位置，即非缺失但可能有噪声的位置)
            denoising_positions = ~target_masks & valid_mask  # 非缺失且有效的位置
            # 只计算denoising位置的均方误差，其他位置保持为0
            if torch.any(denoising_positions):
                squared_diff = (y_pred - y_true) ** 2
                total_losses[denoising_positions] += self.denoising_weight * squared_diff[denoising_positions]
            
            # 只返回有效位置(插补或去噪)的损失值
            active_positions = (imputation_positions | denoising_positions)
            return torch.masked_select(total_losses, active_positions)
        else:
            # 对于reduction='mean'或'sum'，可以分别计算两个损失然后组合
            # 计算插补损失
            imputation_positions = target_masks & valid_mask
            if torch.any(imputation_positions):
                imp_pred = torch.masked_select(y_pred, imputation_positions)
                imp_true = torch.masked_select(y_true, imputation_positions)
                imputation_loss = self.mse_loss(imp_pred, imp_true)
            else:
                imputation_loss = torch.tensor(0.0, device=y_pred.device)
            
            # 计算去噪损失
            denoising_positions = ~target_masks & valid_mask
            if torch.any(denoising_positions):
                denoise_pred = torch.masked_select(y_pred, denoising_positions)
                denoise_true = torch.masked_select(y_true, denoising_positions)
                denoising_loss = self.mse_loss(denoise_pred, denoise_true)
            else:
                denoising_loss = torch.tensor(0.0, device=y_pred.device)
            
            # 根据权重组合两个损失
            total_loss = self.imputation_weight * imputation_loss + self.denoising_weight * denoising_loss
            
            return total_loss


class TrajRLPretrainLoss(nn.Module):
    """
    TrajRL预训练损失函数，实现完整的自监督预训练逻辑
    包括：
    1. 块级轨迹恢复（GPS坐标重建）
    2. 轨迹对比学习（InfoNCE损失）
    3. 联合预训练损失（gamma权重组合）
    
    基于方法-无路网.md中的自监督预训练设计
    """

    def __init__(self, reduction: str = 'mean', disable_mask: bool = False, 
                 gamma: float = 0.5, temperature: float = 0.1, 
                 noise_std: float = 0.001, truncate_ratio: float = 0.1):
        super().__init__()
        
        self.reduction = reduction
        self.disable_mask = disable_mask
        self.gamma = gamma  # 联合损失权重：gamma * L_mask + (1-gamma) * L_cl
        self.temperature = temperature  # 对比学习温度参数
        self.noise_std = noise_std  # 轨迹增强噪声标准差
        self.truncate_ratio = truncate_ratio  # 轨迹截断比例
        
        # 损失函数
        self.mse_loss = nn.MSELoss(reduction=self.reduction)
        
        logger.info(f"TrajRLPretrainLoss - disable_mask: {self.disable_mask}, gamma: {self.gamma}, temperature: {self.temperature}")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                mask: torch.BoolTensor, padding_masks: torch.BoolTensor = None,
                model=None, **kwargs) -> torch.Tensor:
        """
        计算TrajRL预训练损失，包含GPS坐标重建和对比学习
        
        Args:
            y_pred: 模型预测值 (batch_size, seq_len, feat_dim)
            y_true: 真实目标值 (batch_size, seq_len, feat_dim) 
            mask: 特征掩码 (batch_size, seq_len, feat_dim) 1表示需要预测的位置
            padding_masks: 填充掩码 (batch_size, seq_len) 1表示有效位置，0表示填充位置
            model: TrajRLModel实例，用于获取对比学习特征
            
        Returns:
            计算得到的总损失值
        """
        
        # 1. 块级轨迹恢复损失 (L_mask)
        mask_loss = self._compute_mask_loss(y_pred, y_true, mask, padding_masks)
        
        # 2. 轨迹对比学习损失 (L_cl) 
        if model is not None:
            contrastive_loss = self._compute_contrastive_loss(y_true, padding_masks, model)
        else:
            contrastive_loss = torch.tensor(0.0, device=y_pred.device)
            logger.warning("Model参数为None，跳过对比学习损失计算")
        
        # 3. 联合预训练损失 (L_pre = γ * L_mask + (1-γ) * L_cl)
        total_loss = self.gamma * mask_loss + (1 - self.gamma) * contrastive_loss
        
        return total_loss
    
    def _compute_mask_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                          mask: torch.BoolTensor, padding_masks: torch.BoolTensor) -> torch.Tensor:
        """
        计算GPS坐标重建损失
        """
        # 如果disable_mask为True，则直接计算GPS坐标的MSE
        if self.disable_mask:
            # 只计算前2个维度(GPS坐标)的重建损失
            gps_pred = y_pred[:, :, :2]  # (batch_size, seq_len, 2)
            gps_true = y_true[:, :, :2]  # (batch_size, seq_len, 2)
            
            if padding_masks is not None:
                # 应用padding掩码
                valid_mask = padding_masks.unsqueeze(-1).expand_as(gps_pred)
                valid_pred = torch.masked_select(gps_pred, valid_mask)
                valid_true = torch.masked_select(gps_true, valid_mask)
                return self.mse_loss(valid_pred, valid_true)
            else:
                return self.mse_loss(gps_pred, gps_true)
        
        # 准备padding掩码
        if padding_masks is None:
            padding_masks = torch.ones(y_pred.shape[0], y_pred.shape[1], dtype=torch.bool, device=y_pred.device)
        
        # 扩展padding_masks使其维度与特征维度匹配
        valid_mask = padding_masks.unsqueeze(-1).expand_as(y_pred)
        
        # 只关注GPS坐标(前2个维度)的重建
        gps_mask = mask[:, :, :2]  # 只取前2个维度的mask
        gps_valid_mask = valid_mask[:, :, :2]  # 只取前2个维度的valid_mask
        
        # 组合mask：需要预测且在有效范围内的GPS坐标位置
        active_positions = gps_mask & gps_valid_mask
        
        if not torch.any(active_positions):
            # 如果没有需要预测的位置，返回零损失
            return torch.tensor(0.0, device=y_pred.device)
        
        # 提取GPS坐标进行损失计算
        gps_pred = y_pred[:, :, :2]
        gps_true = y_true[:, :, :2]
        
        # 计算masked MSE损失
        masked_pred = torch.masked_select(gps_pred, active_positions)
        masked_true = torch.masked_select(gps_true, active_positions)
        
        return self.mse_loss(masked_pred, masked_true)
    
    def _compute_contrastive_loss(self, trajectories: torch.Tensor, 
                                 padding_masks: torch.BoolTensor, 
                                 model) -> torch.Tensor:
        """
        计算轨迹对比学习损失 (InfoNCE)
        
        Args:
            trajectories: 原始轨迹 (batch_size, seq_len, feat_dim)
            padding_masks: 填充掩码
            model: TrajRLModel实例
            
        Returns:
            对比学习损失
        """
        batch_size = trajectories.size(0)
        device = trajectories.device
        
        # 为每个轨迹创建两个增强视图
        augmented_trajectories = []
        
        for i in range(batch_size):
            traj = trajectories[i:i+1]  # (1, seq_len, feat_dim)
            
            # 创建两个增强视图
            aug1 = model.get_augmented_trajectory(traj, self.noise_std, self.truncate_ratio)
            aug2 = model.get_augmented_trajectory(traj, self.noise_std, self.truncate_ratio)
            
            augmented_trajectories.extend([aug1, aug2])
        
        # 将所有增强轨迹组合成一个批次
        if len(augmented_trajectories) > 0:
            # 处理不同长度的轨迹：填充到相同长度
            max_len = max(traj.size(1) for traj in augmented_trajectories)
            padded_trajectories = []
            aug_padding_masks = []
            
            for traj in augmented_trajectories:
                current_len = traj.size(1)
                if current_len < max_len:
                    # 填充轨迹
                    pad_size = max_len - current_len
                    padded = torch.cat([
                        traj,
                        torch.zeros(1, pad_size, traj.size(2), device=device)
                    ], dim=1)
                    # 创建对应的padding mask
                    mask = torch.cat([
                        torch.ones(1, current_len, device=device),
                        torch.zeros(1, pad_size, device=device)
                    ], dim=1)
                else:
                    padded = traj
                    mask = torch.ones(1, current_len, device=device)
                
                padded_trajectories.append(padded)
                aug_padding_masks.append(mask)
            
            # 组合所有增强轨迹
            all_aug_trajectories = torch.cat(padded_trajectories, dim=0)  # (2*batch_size, max_len, feat_dim)
            all_aug_masks = torch.cat(aug_padding_masks, dim=0)  # (2*batch_size, max_len)
            
            # 获取对比学习特征
            contrastive_features = model.get_contrastive_features(
                all_aug_trajectories, 
                all_aug_masks
            )  # (2*batch_size, hidden_size//4)
            
            # 计算InfoNCE损失
            return self._info_nce_loss(contrastive_features, batch_size)
        else:
            return torch.tensor(0.0, device=device)
    
    def _info_nce_loss(self, features: torch.Tensor, original_batch_size: int) -> torch.Tensor:
        """
        计算InfoNCE对比学习损失
        
        Args:
            features: 对比学习特征 (2*batch_size, feature_dim)
            original_batch_size: 原始批次大小
            
        Returns:
            InfoNCE损失
        """
        device = features.device
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签：每对增强视图为正样本对
        labels = torch.arange(original_batch_size, device=device)
        labels = torch.cat([labels, labels], dim=0)  # (2*batch_size,)
        
        # 计算InfoNCE损失
        total_loss = 0.0
        num_positives = 0
        
        for i in range(2 * original_batch_size):
            # 找到正样本对的索引
            if i < original_batch_size:
                positive_idx = i + original_batch_size  # 第一个增强视图的正样本是对应的第二个增强视图
            else:
                positive_idx = i - original_batch_size  # 第二个增强视图的正样本是对应的第一个增强视图
            
            # 计算分子：正样本对的相似度
            positive_sim = similarity_matrix[i, positive_idx]
            
            # 计算分母：与所有样本的相似度（除了自己）
            mask = torch.ones_like(similarity_matrix[i], dtype=torch.bool)
            mask[i] = False  # 排除自己
            denominator = torch.logsumexp(similarity_matrix[i][mask], dim=0)
            
            # InfoNCE损失
            loss_i = -positive_sim + denominator
            total_loss += loss_i
            num_positives += 1
        
        return total_loss / num_positives if num_positives > 0 else torch.tensor(0.0, device=device)


def mask_length_regularization_loss(model, center_value=0.5, strength=0.01):
    """
    对mask_length_factor应用正则化，鼓励它探索不同的值域
    中心值0.5对应的sigmoid约为0.62，乘以9后得到的mask_length约为6.6
    
    Args:
        model: 包含mask_length_factor的模型
        center_value: 鼓励factor向这个值收敛（对应的sigmoid约为0.62）
        strength: 正则化强度，越大约束越强
        
    Returns:
        正则化损失
    """
    if not hasattr(model, 'mask_length_factor'):
        return 0.0
    
    # 获取当前mask_length_factor值
    factor = model.mask_length_factor
    
    # 计算与理想中心值的偏差，并施加二次惩罚
    # 我们希望在训练过程中factor在合理范围内波动，而不是过快固定
    loss = strength * ((factor - center_value) ** 2)
    
    return loss
