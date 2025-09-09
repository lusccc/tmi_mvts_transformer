import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from tmi.utils import utils
from logzero import logger


# from ..utils import utils


def model_factory(config, data):
    """
    Args:
        data: if task == 'dual_branch_classification', data include trj data and trj feature data
        
    Notes:
        超参数可以通过两种方式传入:
        1. 传统方式: 通过JSON文件路径 (config['trajectory_branch_hyperparams'] = "path/to/hyperparams.json")
        2. 字符串方式: 通过分号分隔的键值对 (config['trajectory_branch_hyperparams'] = "feat_dim=4;max_len=128;d_model=128")
    """
    task = config['task']
    if task in ['imputation_pretrain', 'denoising_pretrain', 'denoising_imputation_pretrain']:
        feat_dim = data.noise_feature_df.shape[1]
        max_seq_len = data.max_seq_len
        model = TSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                     config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                     pos_encoding=config['pos_encoding'], activation=config['activation'],
                                     norm=config['normalization_layer'], freeze=config['freeze'])
        utils.save_model_hyperparams(config, model.hyperparams)
        return model

    if task in ['dual_branch_classification', 'dual_branch_classification_from_scratch']:
        # 使用专门为双分支设计的参数加载方式
        trajectory_hyperparams = utils.load_model_hyperparams(config['trajectory_branch_hyperparams'])
        feature_hyperparams = utils.load_model_hyperparams(config['feature_branch_hyperparams'])
        
        # 创建双分支模型
        model = DualTSTransformerEncoderClassifier(
            trajectory_hyperparams,
            feature_hyperparams,
            len(data.feature_data.class_names), 
            dropout=config['dropout'],
            activation=config['activation']
        )
        
        # 记录日志
        logger.info(f"已创建双分支分类模型，使用TSTransformerEncoderForDualBranch作为分支编码器")
        
        return model
    if task == 'trajectory_branch_classification':
        return TSTransformerEncoderClassifier(**utils.load_model_hyperparams(config['trajectory_branch_hyperparams']),
                                              num_classes=len(data.class_names))
    if task == 'feature_branch_classification':
        return TSTransformerEncoderClassifier(**utils.load_model_hyperparams(config['feature_branch_hyperparams']),
                                              num_classes=len(data.class_names))
    if task in ['feature_branch_classification_from_scratch', 'trajectory_branch_classification_from_scratch']:
        if config['test_only']:
            # 根据task类型加载正确的超参数
            if task == 'feature_branch_classification_from_scratch':
                return TSTransformerEncoderClassifier(**utils.load_model_hyperparams(config['feature_branch_hyperparams']))
            else:  # trajectory_branch_classification_from_scratch
                return TSTransformerEncoderClassifier(**utils.load_model_hyperparams(config['trajectory_branch_hyperparams']))
        else:
            feat_dim = data.noise_feature_df.shape[1]
            max_seq_len = data.max_seq_len
            model = TSTransformerEncoderClassifier(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                                config['num_layers'], config['dim_feedforward'],
                                                num_classes=len(data.class_names), dropout=config['dropout'],
                                                pos_encoding=config['pos_encoding'], activation=config['activation'],
                                                norm=config['normalization_layer'], freeze=config['freeze'])
            utils.save_model_hyperparams(config, model.hyperparams)                                     
            return model
    if 'cnn_classification' in task:
        # return CNN1DClassifier_128()
        return CNN1DClassifier_200()
    if 'lstm_classification' in task:
        return MSRLSTMClassifier()
    if 'dmn_classification' in task:
        return DMNClassifier()
    if task == 'trajrl_classification_from_scratch':
        if config['test_only']:
            # 测试模式，从保存的超参数加载模型
            hyperparams = utils.load_model_hyperparams(config['trajrl_hyperparams'])
            hyperparams['pretrain_mode'] = False  # 确保测试时为分类模式
            return TrajRLModel(**hyperparams)
        else:
            # 训练模式，从头创建模型
            feat_dim = data.noise_feature_df.shape[1]
            model = TrajRLModel(
                feat_dim=feat_dim,
                num_classes=len(data.class_names),
                pretrain_mode=False  # 分类模式
            )
            utils.save_model_hyperparams(config, model.hyperparams)
            return model
    if 'trajrl_classification' in task:
        # TrajRL分类器
        feat_dim = data.noise_feature_df.shape[1]
        return TrajRLModel(
            feat_dim=feat_dim,
            num_classes=len(data.class_names),
            pretrain_mode=False  # 分类模式
        )
    if 'trajrl_pretrain' in task:
        # TrajRL预训练模式
        feat_dim = data.noise_feature_df.shape[1]
        model = TrajRLModel(
            feat_dim=feat_dim,
            num_classes=5,  # 预训练时不重要
            pretrain_mode=True  # 预训练模式
        )
        utils.save_model_hyperparams(config, model.hyperparams)
        return model


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.hyperparams = locals()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation = activation
        self.feat_dim = feat_dim

        # 标准输入投影层
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, feat_dim)
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        
        # 定义可学习的信号增强参数
        self.missing_signal_strength = nn.Parameter(torch.tensor(0.05))
        
        # 定义可学习的mask聚合长度参数
        # 初始值设为0.0，通过映射到更宽区间使模型有更大的调整空间
        # 在forward中将映射到1-10的范围，而非仅1-5
        self.mask_length_factor = nn.Parameter(torch.tensor(0.0))
        
        # 应用自定义权重初始化以提高数值稳定性
        self._init_weights()

    def _init_weights(self):
        """
        使用Xavier均匀初始化对模型权重进行初始化，以提高数值稳定性
        """
        for name, p in self.named_parameters():
            if 'weight' in name and len(p.shape) >= 2:
                # 线性层使用Xavier均匀初始化
                nn.init.xavier_uniform_(p, gain=0.5)
            elif 'bias' in name:
                # 偏置项初始化为0
                nn.init.zeros_(p)
        
        # 特别处理多头注意力层的权重，使用更保守的初始化
        for layer in self.transformer_encoder.layers:
            if hasattr(layer, 'self_attn'):
                # 对注意力权重使用更小的初始值范围
                for name, p in layer.self_attn.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(p, gain=0.2)
                    elif 'bias' in name:
                        nn.init.zeros_(p)
                        
        # 对输出层使用更小的初始化范围
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def create_custom_attention_mask(self, feature_masks, padding_masks):
        """
        创建自定义注意力掩码，并支持自适应mask长度
        
        Args:
            feature_masks: (batch_size, seq_length, feat_dim) 特征掩码，1表示需要预测的位置(即缺失位置)
            padding_masks: (batch_size, seq_length) 填充掩码，1表示有效位置，0表示填充位置
            
        Returns:
            attn_mask: (seq_length, seq_length) 全局注意力掩码
            missing_stats: 字典，包含缺失特征的统计信息
        """
        batch_size, seq_length, feat_dim = feature_masks.shape
        device = feature_masks.device
        
        # 1. 确定每个时间步是否有需要预测的特征(缺失值) - (batch_size, seq_length)
        has_missing = torch.any(feature_masks, dim=2)
        
        # 2. 创建标准因果掩码 (所有批次共用)
        causal_mask = torch.ones(seq_length, seq_length, device=device).triu(diagonal=1)
        
        # 3. 使用全局掩码 + 输入信号增强的方式
        # 将因果掩码转换为float，并将1替换为一个非常大但有限的负数
        attn_mask = causal_mask.float() * -1e9
        
        # 4. 生成缺失特征的统计信息，用于在forward方法中增强输入
        missing_stats = {
            'has_missing': has_missing,
            'missing_ratio': has_missing.float().mean().item(),
            'causal_only': causal_mask
        }
        
        return attn_mask, missing_stats

    def forward(self, X, padding_masks, feature_masks=None):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) 原始特征输入
            padding_masks: (batch_size, seq_length) 填充掩码，1表示保留该位置，0表示填充位置
            feature_masks: (batch_size, seq_length, feat_dim) 特征掩码，1表示需要预测的位置(即缺失位置)
            
        Returns:
            output: (batch_size, seq_length, feat_dim) 输出特征
        """
        batch_size, seq_length, feat_dim = X.shape
        device = X.device
        
        # 基本投影：将输入从feat_dim投影到d_model
        inp = X.permute(1, 0, 2)  # (seq_length, batch_size, feat_dim)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # (seq_length, batch_size, d_model)
        
        # 添加位置编码
        inp = self.pos_enc(inp)  # (seq_length, batch_size, d_model)
        
        # 创建注意力掩码
        attn_mask = None
        
        # 处理feature_masks以创建自定义注意力掩码
        if feature_masks is not None:
            # 计算自适应mask长度 (将mask_length_factor映射到合理范围)
            # 直接在forward方法中计算，确保计算图连接正确
            mask_length = 1.0 + 9.0 * torch.sigmoid(self.mask_length_factor)  # 映射到1-10范围内
            
            # 使用优化后的掩码创建方法
            attn_mask, missing_stats = self.create_custom_attention_mask(feature_masks, padding_masks)
            
            # 使用向量化操作高效地增强输入
            if torch.any(feature_masks):
                # 获取时间步级别的缺失信息
                has_missing = missing_stats['has_missing']  # (batch_size, seq_length)
                
                # 将信息转换到正确的形状并添加到输入中
                missing_signal = has_missing.permute(1, 0).unsqueeze(2).float()
                
                # 创建增强信号并广播
                clamped_strength = torch.clamp(self.missing_signal_strength, 0.0, 0.5)
                enhancement = torch.zeros_like(inp)
                enhancement = enhancement + missing_signal * clamped_strength
                
                # 计算基础增强信号
                base_enhancement = enhancement.clone()
                
                # 应用自适应mask长度：将缺失信号向周围时间步传播
                # 只有当mask_length > 1.0时才进行传播
                if mask_length > 1.0:
                    # 为了数值稳定性，使用detach复制当前mask_length值仅用于控制流
                    # 但计算权重时仍使用原始mask_length保持计算图
                    mask_length_value = mask_length.detach().item()
                    spread_steps = max(1, int(round(mask_length_value - 1)))
                    
                    # mask长度影响因子 - 随mask_length增大而增强 (保持梯度流)
                    # 这将使更长的mask产生更强的影响，增强梯度信号
                    mask_impact = 0.5 + 0.5 * (mask_length - 1.0) / 9.0  # 随着mask_length从1到10变化，影响从0.5到1.0
                    
                    # 创建扩散权重，越靠近缺失位置权重越大
                    for step in range(1, spread_steps + 1):
                        # 计算权重，直接使用mask_length参与计算，保持梯度链接
                        # 归一化步骤，确保步骤在0-1范围内
                        step_ratio = torch.tensor(step, dtype=torch.float, device=device) / (mask_length - 1.0 + 1e-6)
                        # 线性衰减权重，添加mask_impact因子增强梯度
                        weight_scale = (1.0 - step_ratio) * (mask_length > 1.0).float() * mask_impact
                        
                        # 向前传播
                        if step < seq_length:
                            forward_weight = clamped_strength * weight_scale
                            shifted_signal = torch.cat([
                                missing_signal[step:], 
                                torch.zeros(step, batch_size, 1, device=device)
                            ], dim=0)
                            enhancement = enhancement + shifted_signal * forward_weight
                            
                        # 向后传播
                        if step < seq_length:
                            backward_weight = clamped_strength * weight_scale
                            shifted_signal = torch.cat([
                                torch.zeros(step, batch_size, 1, device=device),
                                missing_signal[:-step] if step < seq_length else torch.zeros(0, batch_size, 1, device=device)
                            ], dim=0)
                            enhancement = enhancement + shifted_signal * backward_weight
                
                # 将增强信号添加到原始输入
                inp = inp + enhancement
                
                # 保存当前的mask_length值用于监控
                missing_stats['mask_length'] = mask_length.detach().item()
        
        # 应用Transformer编码器
        # 使用padding_masks来屏蔽填充位置
        output = self.transformer_encoder(inp, 
                                        #  mask=attn_mask,  # TODO 并没有使用因果掩码矩阵！！！
                                         src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        
        output = self.act(output)  # 应用激活函数
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassifier(nn.Module):
    """
    通用的Transformer编码器分类器基类，可用于单分支分类任务。
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassifier, self).__init__()

        self.hyperparams = locals()
        
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        # 标准输入投影层
        self.project_inp = nn.Linear(feat_dim, d_model)
        
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)
        
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                  activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                           dropout * (1.0 - freeze), activation=activation)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类层，不同于TSTransformerEncoder的输出层
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)
        
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        
        # 定义可学习的信号增强参数
        self.missing_signal_strength = nn.Parameter(torch.tensor(0.05))
        
        # 创建TSTransformerEncoder实例以复用其create_custom_attention_mask方法
        self.encoder_helper = TSTransformerEncoder(
            feat_dim=feat_dim, 
            max_len=max_len, 
            d_model=d_model, 
            n_heads=n_heads, 
            num_layers=0,  # 不需要实际的层
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            pos_encoding=pos_encoding,
            activation=activation,
            norm=norm,
            freeze=freeze
        )

    def build_output_module(self, d_model, max_len, num_classes):
        """构建分类输出模块，将编码器的输出映射到类别概率"""
        # 计算输入特征大小 - 扁平化的d_model特征
        input_size = max_len * d_model
        
        # 创建分类层
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, X, padding_masks, feature_masks=None):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) 原始特征输入
            padding_masks: (batch_size, seq_length) 填充掩码，1表示保留该位置，0表示填充位置 
            feature_masks: (batch_size, seq_length, feat_dim) 特征掩码，1表示需要预测的位置(即缺失位置)
            
        Returns:
            output: (batch_size, d_output)
        """
        batch_size, seq_length, feat_dim = X.shape
        device = X.device
        
        # 基本投影：将输入从feat_dim投影到d_model
        inp = X.permute(1, 0, 2)  # (seq_length, batch_size, feat_dim)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # (seq_length, batch_size, d_model)
        
        # 添加位置编码
        inp = self.pos_enc(inp)  # (seq_length, batch_size, d_model)
        
        # 创建注意力掩码
        attn_mask = None
        
        # 处理feature_masks以创建自定义注意力掩码
        if feature_masks is not None:
            # 计算自适应mask长度，确保梯度传播
            mask_length = 1.0 + 9.0 * torch.sigmoid(self.missing_signal_strength.new_tensor(self.encoder_helper.mask_length_factor))
            
            # 使用优化后的掩码创建方法
            attn_mask, missing_stats = self.encoder_helper.create_custom_attention_mask(feature_masks, padding_masks)
            
            # 使用向量化操作高效地增强输入
            if torch.any(feature_masks):
                # 获取时间步级别的缺失信息
                has_missing = missing_stats['has_missing']  # (batch_size, seq_length)
                
                # 将信息转换到正确的形状并添加到输入中
                missing_signal = has_missing.permute(1, 0).unsqueeze(2).float()
                
                # 创建增强信号并广播
                clamped_strength = torch.clamp(self.missing_signal_strength, 0.0, 0.5)
                enhancement = torch.zeros_like(inp)
                enhancement = enhancement + missing_signal * clamped_strength
                
                # 计算基础增强信号
                base_enhancement = enhancement.clone()
                
                # 应用自适应mask长度：将缺失信号向周围时间步传播
                # 只有当mask_length > 1.0时才进行传播
                if mask_length > 1.0:
                    # 为了数值稳定性，使用detach复制当前mask_length值仅用于控制流
                    # 但计算权重时仍使用原始mask_length保持计算图
                    mask_length_value = mask_length.detach().item()
                    spread_steps = max(1, int(round(mask_length_value - 1)))
                    
                    # mask长度影响因子 - 随mask_length增大而增强 (保持梯度流)
                    # 这将使更长的mask产生更强的影响，增强梯度信号
                    mask_impact = 0.5 + 0.5 * (mask_length - 1.0) / 9.0  # 随着mask_length从1到10变化，影响从0.5到1.0
                    
                    # 创建扩散权重，越靠近缺失位置权重越大
                    for step in range(1, spread_steps + 1):
                        # 计算权重，直接使用mask_length参与计算，保持梯度链接
                        # 归一化步骤，确保步骤在0-1范围内
                        step_ratio = torch.tensor(step, dtype=torch.float, device=device) / (mask_length - 1.0 + 1e-6)
                        # 线性衰减权重，添加mask_impact因子增强梯度
                        weight_scale = (1.0 - step_ratio) * (mask_length > 1.0).float() * mask_impact
                        
                        # 向前传播
                        if step < seq_length:
                            forward_weight = clamped_strength * weight_scale
                            shifted_signal = torch.cat([
                                missing_signal[step:], 
                                torch.zeros(step, batch_size, 1, device=device)
                            ], dim=0)
                            enhancement = enhancement + shifted_signal * forward_weight
                            
                        # 向后传播
                        if step < seq_length:
                            backward_weight = clamped_strength * weight_scale
                            shifted_signal = torch.cat([
                                torch.zeros(step, batch_size, 1, device=device),
                                missing_signal[:-step] if step < seq_length else torch.zeros(0, batch_size, 1, device=device)
                            ], dim=0)
                            enhancement = enhancement + shifted_signal * backward_weight
                
                # 将增强信号添加到原始输入
                inp = inp + enhancement
        
        # 应用Transformer编码器
        output = self.transformer_encoder(inp, 
                                        mask=attn_mask,
                                        src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        
        output = self.act(output)  # 应用激活函数
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        
        # 应用掩码以忽略填充位置
        masked_output = output * padding_masks.unsqueeze(-1)  # 将填充位置置零
        
        # 扁平化特征
        flattened = masked_output.reshape(masked_output.shape[0], -1)  # (batch_size, seq_length * d_model)
        
        # 应用分类层
        output = self.output_layer(flattened)  # (batch_size, num_classes)
        
        return output


class TSTransformerEncoderForDualBranch(TSTransformerEncoder):
    """
    专为双分支模型设计的Transformer编码器，输出扁平化的特征表示。
    继承自TSTransformerEncoder以重用基础功能。
    """
    
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderForDualBranch, self).__init__(
            feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, 
            dropout, pos_encoding, activation, norm, freeze
        )
        
        # 移除原始TSTransformerEncoder的输出层，直接使用扁平化
        delattr(self, 'output_layer')
        
        # 确保继承了可学习的信号增强参数
        # self.missing_signal_strength在父类已定义

    def forward(self, X, padding_masks, feature_masks=None):
        """
        前向传播函数，输出扁平化的特征表示
        
        Args:
            X: (batch_size, seq_length, feat_dim) 原始特征输入
            padding_masks: (batch_size, seq_length) 填充掩码，1表示保留该位置，0表示填充位置
            feature_masks: (batch_size, seq_length, feat_dim) 特征掩码，1表示需要预测的位置(即缺失位置)
            
        Returns:
            output: (batch_size, seq_length * d_model) 扁平化的特征表示
        """
        batch_size, seq_length, feat_dim = X.shape
        device = X.device
        
        # 基本投影：将输入从feat_dim投影到d_model
        inp = X.permute(1, 0, 2)  # (seq_length, batch_size, feat_dim)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # (seq_length, batch_size, d_model)
        
        # 添加位置编码
        inp = self.pos_enc(inp)  # (seq_length, batch_size, d_model)
        
        # 创建注意力掩码
        attn_mask = None
        
        # 处理feature_masks以创建自定义注意力掩码
        if feature_masks is not None:
            # 计算自适应mask长度，直接在forward方法中计算以确保计算图连接
            mask_length = 1.0 + 9.0 * torch.sigmoid(self.mask_length_factor)
            
            # 使用优化后的掩码创建方法
            attn_mask, missing_stats = self.create_custom_attention_mask(feature_masks, padding_masks)
            
            # 使用向量化操作高效地增强输入
            if torch.any(feature_masks):
                # 获取时间步级别的缺失信息
                has_missing = missing_stats['has_missing']  # (batch_size, seq_length)
                
                # 将信息转换到正确的形状并添加到输入中
                missing_signal = has_missing.permute(1, 0).unsqueeze(2).float()
                
                # 创建增强信号并广播
                clamped_strength = torch.clamp(self.missing_signal_strength, 0.0, 0.5)
                enhancement = torch.zeros_like(inp)
                enhancement = enhancement + missing_signal * clamped_strength
                
                # 计算基础增强信号
                base_enhancement = enhancement.clone()
                
                # 应用自适应mask长度：将缺失信号向周围时间步传播
                # 只有当mask_length > 1.0时才进行传播
                if mask_length > 1.0:
                    # 为了数值稳定性，使用detach复制当前mask_length值仅用于控制流
                    # 但计算权重时仍使用原始mask_length保持计算图
                    mask_length_value = mask_length.detach().item()
                    spread_steps = max(1, int(round(mask_length_value - 1)))
                    
                    # mask长度影响因子 - 随mask_length增大而增强 (保持梯度流)
                    # 这将使更长的mask产生更强的影响，增强梯度信号
                    mask_impact = 0.5 + 0.5 * (mask_length - 1.0) / 9.0  # 随着mask_length从1到10变化，影响从0.5到1.0
                    
                    # 创建扩散权重，越靠近缺失位置权重越大
                    for step in range(1, spread_steps + 1):
                        # 计算权重，直接使用mask_length参与计算，保持梯度链接
                        # 归一化步骤，确保步骤在0-1范围内
                        step_ratio = torch.tensor(step, dtype=torch.float, device=device) / (mask_length - 1.0 + 1e-6)
                        # 线性衰减权重，添加mask_impact因子增强梯度
                        weight_scale = (1.0 - step_ratio) * (mask_length > 1.0).float() * mask_impact
                        
                        # 向前传播
                        if step < seq_length:
                            forward_weight = clamped_strength * weight_scale
                            shifted_signal = torch.cat([
                                missing_signal[step:], 
                                torch.zeros(step, batch_size, 1, device=device)
                            ], dim=0)
                            enhancement = enhancement + shifted_signal * forward_weight
                            
                        # 向后传播
                        if step < seq_length:
                            backward_weight = clamped_strength * weight_scale
                            shifted_signal = torch.cat([
                                torch.zeros(step, batch_size, 1, device=device),
                                missing_signal[:-step] if step < seq_length else torch.zeros(0, batch_size, 1, device=device)
                            ], dim=0)
                            enhancement = enhancement + shifted_signal * backward_weight
                
                # 将增强信号添加到原始输入
                inp = inp + enhancement
        
        # 应用Transformer编码器
        output = self.transformer_encoder(inp, 
                                        mask=attn_mask,
                                        src_key_padding_mask=~padding_masks)
        output = self.act(output)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        
        # 应用掩码并扁平化
        masked_output = output * padding_masks.unsqueeze(-1)  # 将填充位置置零
        flattened = masked_output.reshape(masked_output.shape[0], -1)  # (batch_size, seq_length * d_model)
        
        return flattened


class DualTSTransformerEncoderClassifier(nn.Module):
    """
    双分支Transformer编码器分类器，结合轨迹和特征两个分支的信息。
    """
    def __init__(self, trajectory_branch_hyperparams, feature_branch_hyperparams, num_classes, dropout=0.1,
                 activation='gelu'):
        super(DualTSTransformerEncoderClassifier, self).__init__()
        self.num_classes = num_classes
        
        # 记录原始特征维度和序列长度，用于后续处理
        self.trajectory_feat_dim = trajectory_branch_hyperparams.get('feat_dim')
        self.feature_feat_dim = feature_branch_hyperparams.get('feat_dim')
        self.trajectory_max_len = trajectory_branch_hyperparams.get('max_len')
        self.feature_max_len = feature_branch_hyperparams.get('max_len')
        
        # 创建分支模型，使用专门为双分支设计的TSTransformerEncoderForDualBranch
        self.trajectory_branch = TSTransformerEncoderForDualBranch(**trajectory_branch_hyperparams)
        self.feature_branch = TSTransformerEncoderForDualBranch(**feature_branch_hyperparams)
        
        # 保存d_model，用于维度调整
        self.trajectory_d_model = self.trajectory_branch.d_model
        self.feature_d_model = self.feature_branch.d_model
        
        # 计算合并后的特征维度
        total_channels = self.trajectory_d_model + self.feature_d_model
        
        # 卷积融合网络
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=total_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 全局池化层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 分类层
        fc_input_dim = 64 * 2  # 全局平均池化 + 全局最大池化
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, X1, padding_mask1, X2, padding_mask2):
        """
        双分支转换器分类器的前向传播
        
        Args:
            X1: 轨迹分支输入，形状 (batch_size, seq_length, feat_dim) 或 (batch_size, seq_length, feat_dim*2)
            padding_mask1: 轨迹分支填充掩码，形状 (batch_size, seq_length)
            X2: 特征分支输入，形状 (batch_size, seq_length, feat_dim) 或 (batch_size, seq_length, feat_dim*2)
            padding_mask2: 特征分支填充掩码，形状 (batch_size, seq_length)
            
        Returns:
            output: 分类结果，形状 (batch_size, num_classes)
        """
        batch_size = X1.size(0)
        
        # 获取两个分支的特征表示
        trajectory_output = self.trajectory_branch(X1, padding_mask1)  # (batch_size, seq_len*d_model)
        feature_output = self.feature_branch(X2, padding_mask2)  # (batch_size, seq_len*d_model)
        
        # 重塑特征为三维张量，用于后续处理
        trajectory_output = trajectory_output.view(batch_size, self.trajectory_max_len, self.trajectory_d_model)
        feature_output = feature_output.view(batch_size, self.feature_max_len, self.feature_d_model)
        
        # 在特征维度上拼接
        combined_output = torch.cat([trajectory_output, feature_output], dim=2)
        
        # 转换维度用于卷积操作
        combined_output = combined_output.permute(0, 2, 1)  # (batch_size, channels, seq_len)
        
        # 应用卷积层
        conv_output = self.conv_layers(combined_output)  # (batch_size, 64, seq_len/2)
        
        # 应用全局池化
        avg_pooled = self.global_avg_pool(conv_output).squeeze(-1)  # (batch_size, 64)
        max_pooled = self.global_max_pool(conv_output).squeeze(-1)  # (batch_size, 64)
        
        # 融合池化结果
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # (batch_size, 128)
        
        # 分类
        output = self.classifier(pooled_features)  # (batch_size, num_classes)
        
        return output


class LambdaLayer(nn.Module):
    def __init__(self):
        super(LambdaLayer, self).__init__()

    def forward(self, x):
        return x

# #########################################################
# ############# below are COMPARATIVE MODELS: #############
# #########################################################

# ****** TrajRL Model (without road network dependency) ******

class GPSPointEncoder(nn.Module):
    """GPS点编码器，将经纬度坐标编码为特征向量"""
    
    def __init__(self, d_model=128):
        super(GPSPointEncoder, self).__init__()
        self.d_model = d_model
        # 将经纬度坐标编码为高维特征
        self.spatial_encoder = nn.Linear(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, coords):
        """
        Args:
            coords: (batch_size, seq_length, 2) GPS坐标 [lat, lon]
        Returns:
            spatial_emb: (batch_size, seq_length, d_model) 空间特征嵌入
        """
        spatial_emb = self.spatial_encoder(coords)  # (batch_size, seq_length, d_model)
        spatial_emb = self.layer_norm(spatial_emb)
        return spatial_emb


class MultiAspectTimeEncoder(nn.Module):
    """多面时间编码器，包含周期编码器和间隔编码器"""
    
    def __init__(self, d_model=128, max_seq_len=128):
        super(MultiAspectTimeEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 时间周期编码器参数
        self.day_emb = nn.Embedding(7, d_model // 4)  # 一周7天
        self.hour_emb = nn.Embedding(24, d_model // 4)  # 一天24小时
        self.minute_emb = nn.Embedding(60, d_model // 4)  # 一小时60分钟
        
        # 时间间隔编码器
        self.interval_scales = [1/15, 1/60, 1/150]  # 对应{15,60,150}
        self.interval_encoder = nn.Linear(len(self.interval_scales), d_model // 4)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, timestamps):
        """
        Args:
            timestamps: (batch_size, seq_length) 时间戳序列
        Returns:
            time_emb: (batch_size, seq_length, d_model) 时间特征嵌入
        """
        batch_size, seq_length = timestamps.shape
        device = timestamps.device
        
        # 简化处理：直接使用位置索引而不是复杂的时间解析
        # 避免索引越界问题
        position_indices = torch.arange(seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        # 使用位置作为时间代理，确保索引在有效范围内
        days = (position_indices % 7).long()  # 模拟一周循环
        hours = (position_indices % 24).long()  # 模拟一天循环  
        minutes = (position_indices % 60).long()  # 模拟一小时循环
        
        # 周期编码 - 确保索引不会越界
        day_emb = self.day_emb(days)  # (batch_size, seq_length, d_model//4)
        hour_emb = self.hour_emb(hours)
        minute_emb = self.minute_emb(minutes)
        
        # 时间间隔编码 - 简化处理
        # 使用位置差作为间隔代理，避免复杂的时间戳处理
        intervals = torch.ones(batch_size, seq_length, device=device)  # 统一间隔
        intervals[:, 0] = 0  # 第一个位置没有间隔
        
        # 多尺度时间间隔特征
        scaled_intervals = []
        for scale in self.interval_scales:
            scaled_intervals.append((intervals * scale).unsqueeze(-1))
        
        interval_features = torch.cat(scaled_intervals, dim=-1)  # (batch_size, seq_length, len(scales))
        interval_emb = self.interval_encoder(interval_features)  # (batch_size, seq_length, d_model//4)
        
        # 拼接所有时间特征
        time_emb = torch.cat([day_emb, hour_emb, minute_emb, interval_emb], dim=-1)
        time_emb = self.layer_norm(time_emb)
        
        return time_emb


class InPatchEncoder(nn.Module):
    """块内编码器，使用GRU对块内GPS点序列进行编码"""
    
    def __init__(self, d_model=128, hidden_size=64):
        super(InPatchEncoder, self).__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.gru = nn.GRU(d_model, hidden_size, batch_first=True)
        
    def forward(self, patch_embeddings):
        """
        Args:
            patch_embeddings: (batch_size, patch_length, d_model) 块内点的嵌入
        Returns:
            patch_repr: (batch_size, hidden_size) 块表示
        """
        gru_out, _ = self.gru(patch_embeddings)  # (batch_size, patch_length, hidden_size)
        # 使用平均池化得到块表示
        patch_repr = torch.mean(gru_out, dim=1)  # (batch_size, hidden_size)
        return patch_repr


class CrossPatchEncoder(nn.Module):
    """跨块编码器，使用Transformer对块序列进行编码"""
    
    def __init__(self, d_model=64, n_heads=4, num_layers=2, dim_feedforward=256):
        super(CrossPatchEncoder, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, patch_sequence):
        """
        Args:
            patch_sequence: (batch_size, num_patches, d_model) 块序列
        Returns:
            encoded_patches: (batch_size, num_patches, d_model) 编码后的块序列
        """
        return self.transformer(patch_sequence)


class MultiScaleTrajectoryEncoder(nn.Module):
    """多尺度轨迹编码器，结合InPatch和CrossPatch编码器"""
    
    def __init__(self, d_model=128, patch_size=8, hidden_size=64, n_heads=4, num_layers=2):
        super(MultiScaleTrajectoryEncoder, self).__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # InPatch编码器
        self.inpatch_encoder = InPatchEncoder(d_model, hidden_size)
        
        # CrossPatch编码器
        self.crosspatch_encoder = CrossPatchEncoder(hidden_size, n_heads, num_layers)
        
        # 多尺度融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3))  # 3个尺度的权重
        
    def forward(self, trajectory_embeddings):
        """
        Args:
            trajectory_embeddings: (batch_size, seq_length, d_model) 轨迹点嵌入序列
        Returns:
            trajectory_repr: (batch_size, hidden_size) 轨迹表示
        """
        batch_size, seq_length, d_model = trajectory_embeddings.shape
        
        # 计算块数量
        num_patches = seq_length // self.patch_size
        if num_patches == 0:
            num_patches = 1
            effective_patch_size = seq_length
        else:
            effective_patch_size = self.patch_size
        
        # 分块处理
        patch_reprs = []
        for i in range(num_patches):
            start_idx = i * effective_patch_size
            end_idx = min((i + 1) * effective_patch_size, seq_length)
            patch = trajectory_embeddings[:, start_idx:end_idx, :]  # (batch_size, patch_length, d_model)
            
            # InPatch编码
            patch_repr = self.inpatch_encoder(patch)  # (batch_size, hidden_size)
            patch_reprs.append(patch_repr)
        
        # 构建块序列
        patch_sequence = torch.stack(patch_reprs, dim=1)  # (batch_size, num_patches, hidden_size)
        
        # CrossPatch编码
        encoded_patches = self.crosspatch_encoder(patch_sequence)  # (batch_size, num_patches, hidden_size)
        
        # 多尺度融合（简化版本）
        # 这里实现简单的平均池化，实际可以扩展为多个尺度
        trajectory_repr = torch.mean(encoded_patches, dim=1)  # (batch_size, hidden_size)
        
        return trajectory_repr


class TrajRLModel(nn.Module):
    """
    TrajRL模型，去除路网依赖
    基于方法-无路网.md中的方法实现
    """
    
    def __init__(self, feat_dim=4, d_model=256, patch_size=4, hidden_size=128, 
                 n_heads=16, num_layers=3, num_classes=5, dropout=0.1,
                 pretrain_mode=False):
        super(TrajRLModel, self).__init__()
        
        # 保存超参数
        self.hyperparams = locals()
        
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.pretrain_mode = pretrain_mode
        self.hidden_size = hidden_size
        
        # GPS点编码器 - 将前2个特征(lat, lon)编码
        self.gps_encoder = GPSPointEncoder(d_model // 2)
        
        # 多面时间编码器 - 假设第3个特征是时间
        self.time_encoder = MultiAspectTimeEncoder(d_model // 2)
        
        # 多尺度轨迹编码器
        self.trajectory_encoder = MultiScaleTrajectoryEncoder(
            d_model=d_model, 
            patch_size=patch_size, 
            hidden_size=hidden_size,
            n_heads=n_heads,
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 预训练相关组件
        self.coord_reconstruction_head = nn.Linear(hidden_size, 2)  # GPS坐标重建头
        self.projection_head = nn.Sequential(  # 对比学习投影头
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        
    def forward(self, x, padding_masks=None, feature_masks=None, **kwargs):
        """
        前向传播，支持预训练和微调模式
        
        Args:
            x: (batch_size, seq_length, feat_dim) 输入轨迹
            padding_masks: (batch_size, seq_length) 填充掩码，1表示有效，0表示填充
            feature_masks: (batch_size, seq_length, feat_dim) 特征掩码，用于预训练
            
        Returns:
            预训练模式：返回重建的GPS坐标
            分类模式：返回分类logits
        """
        batch_size, seq_length, feat_dim = x.shape
        
        # 提取GPS坐标 (前2个特征) 和时间 (第3个特征)
        gps_coords = x[:, :, :2]  # (batch_size, seq_length, 2)
        
        # 简化处理：使用第3个特征作为时间，如果没有则用位置索引
        if feat_dim > 2:
            timestamps = x[:, :, 2]  # (batch_size, seq_length)
        else:
            # 如果没有时间特征，使用位置索引作为代理
            timestamps = torch.arange(seq_length, device=x.device).float().unsqueeze(0).repeat(batch_size, 1)
        
        # GPS点编码
        gps_embeddings = self.gps_encoder(gps_coords)  # (batch_size, seq_length, d_model//2)
        
        # 时间编码
        time_embeddings = self.time_encoder(timestamps)  # (batch_size, seq_length, d_model//2)
        
        # 拼接GPS和时间嵌入
        combined_embeddings = torch.cat([gps_embeddings, time_embeddings], dim=-1)  # (batch_size, seq_length, d_model)
        
        # 添加位置编码
        if seq_length <= self.pos_encoding.size(1):
            combined_embeddings = combined_embeddings + self.pos_encoding[:, :seq_length, :]
        
        # 多尺度轨迹编码
        trajectory_features = self.trajectory_encoder(combined_embeddings)  # (batch_size, hidden_size)
        
        # 根据模式返回不同结果
        if self.pretrain_mode:
            # 预训练模式：返回重建的轨迹特征，与输入格式一致
            # 对于GPS坐标重建，我们需要重建前2个维度的特征
            coord_reconstruction = self.coord_reconstruction_head(trajectory_features)  # (batch_size, 2)
            
            # 构建完整的重建输出，保持与输入相同的形状
            if feat_dim == 2:
                # 如果输入只有GPS坐标，直接广播
                reconstruction = coord_reconstruction.unsqueeze(1).repeat(1, seq_length, 1)
            else:
                # 如果输入有更多特征，重建GPS坐标，其他特征保持原值
                reconstruction = x.clone()  # 复制原始输入
                coord_recon_expanded = coord_reconstruction.unsqueeze(1).repeat(1, seq_length, 1)
                reconstruction[:, :, :2] = coord_recon_expanded  # 只替换前2个维度(GPS坐标)
            
            return reconstruction
        else:
            # 分类模式：返回分类结果
            logits = self.classifier(trajectory_features)
            return logits
    
    def get_contrastive_features(self, x, padding_masks=None, **kwargs):
        """
        获取对比学习特征，用于自监督预训练
        
        Args:
            x: (batch_size, seq_length, feat_dim) 输入轨迹
            padding_masks: (batch_size, seq_length) 填充掩码
            
        Returns:
            contrastive_features: (batch_size, hidden_size//4) 对比学习特征
        """
        batch_size, seq_length, feat_dim = x.shape
        
        # 提取GPS坐标和时间特征（与forward相同的处理）
        gps_coords = x[:, :, :2]
        if feat_dim > 2:
            timestamps = x[:, :, 2]
        else:
            timestamps = torch.arange(seq_length, device=x.device).float().unsqueeze(0).repeat(batch_size, 1)
        
        # 编码
        gps_embeddings = self.gps_encoder(gps_coords)
        time_embeddings = self.time_encoder(timestamps)
        combined_embeddings = torch.cat([gps_embeddings, time_embeddings], dim=-1)
        
        if seq_length <= self.pos_encoding.size(1):
            combined_embeddings = combined_embeddings + self.pos_encoding[:, :seq_length, :]
        
        # 轨迹编码
        trajectory_features = self.trajectory_encoder(combined_embeddings)
        
        # 对比学习特征投影
        contrastive_features = self.projection_head(trajectory_features)
        return contrastive_features
    
    def get_augmented_trajectory(self, trajectory, noise_std=0.001, truncate_ratio=0.1):
        """
        轨迹增强方法，用于对比学习
        
        Args:
            trajectory: (batch_size, seq_length, feat_dim) 原始轨迹
            noise_std: 坐标噪声标准差
            truncate_ratio: 截断比例
            
        Returns:
            augmented_trajectory: 增强后的轨迹
        """
        augmented = trajectory.clone()
        
        # 1. 坐标扰动 - 只对GPS坐标(前2个维度)添加噪声
        gps_noise = torch.randn_like(trajectory[:, :, :2]) * noise_std
        augmented[:, :, :2] = augmented[:, :, :2] + gps_noise
        
        # 2. 轨迹截断 (随机选择前缀或后缀)
        seq_length = trajectory.size(1)
        truncate_length = int(seq_length * truncate_ratio)
        
        if truncate_length > 0 and truncate_length < seq_length:
            if torch.rand(1).item() > 0.5:
                # 截取前缀
                augmented = augmented[:, :-truncate_length, :]
            else:
                # 截取后缀  
                augmented = augmented[:, truncate_length:, :]
        
        return augmented
    
    def set_pretrain_mode(self, pretrain_mode=True):
        """设置预训练模式"""
        self.pretrain_mode = pretrain_mode


# ****** CNN1DClassifier_200 ******

class CNN1DClassifier_200(nn.Module):
    """
    reproduction code for paper `Inferring transportation modes from GPS trajectories using a convolutional neural network'
    https://github.com/sinadabiri/Transport-Mode-GPS-CNN
    """

    def __init__(self):
        super().__init__()
        self.num_classes = 5

        # 200xn_ori_channels => 200x32
        self.conv_1 = nn.Conv1d(in_channels=4,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                # (1*(200-1) - 200 + 3) / 2 = 1
                                padding=1)
        #  200x32 => 200x32
        self.conv_2 = nn.Conv1d(in_channels=32,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                # (1*(200-1) - 200 + 3) / 2 = 1
                                padding=1)
        # 200x32 => 100x32
        self.pool_1 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(100-1) - 200 + 2) / 2 = 0
                                   padding=0)
        # 100x100x32 => 100x100x64
        self.conv_3 = nn.Conv1d(in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                # (1*(100-1) - 100 + 3) / 2 = 1
                                padding=1)
        # 100x100x64 => 100x100x64
        self.conv_4 = nn.Conv1d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                # (1*(100-1) - 100 + 3) / 2 = 1
                                padding=1)
        # 100x64 => 50x64
        self.pool_2 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(50-1) - 100 + 2) / 2 = 0
                                   padding=0)
        # 50x64 => 50x128
        self.conv_5 = nn.Conv1d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                # (1*(50-1) - 50 + 3) / 2 = 1
                                padding=1)
        # 50x128 => 50x128
        self.conv_6 = nn.Conv1d(in_channels=128,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                # (1*(50-1) - 50 + 3) / 2 = 1
                                padding=1)
        # 50x128 => 25x128
        self.pool_3 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(25-1) - 50 + 2) / 2 = 0
                                   padding=0)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(.5)
        self.dense = nn.Linear(25 * 128, self.num_classes)

    def forward(self, x, *args, **kwargs):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = F.relu(x)
        x = self.pool_2(x)
        x = self.conv_5(x)
        x = F.relu(x)
        x = self.conv_6(x)
        x = F.relu(x)
        x = self.pool_3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


# ****** CNN1DClassifier_128 ******

class CNN1DClassifier_128(nn.Module):
    """
    reproduction code for paper `Inferring transportation modes from GPS trajectories using a convolutional neural network'
    https://github.com/sinadabiri/Transport-Mode-GPS-CNN
    """

    def __init__(self):
        super().__init__()
        self.num_classes = 5

        # 128xn_ori_channels => 128x32
        self.conv_1 = nn.Conv1d(in_channels=4,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                # (1*(128-1) - 128 + 3) / 2 = 1
                                padding=1)
        #  128x32 => 128x32
        self.conv_2 = nn.Conv1d(in_channels=32,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                # (1*(128-1) - 128 + 3) / 2 = 1
                                padding=1)
        # 128x32 => 64x32
        self.pool_1 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(64-1) - 128 + 2) / 2 = 0
                                   padding=0)
        # 64x64x32 => 64x64x64
        self.conv_3 = nn.Conv1d(in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                # (1*(64-1) - 64 + 3) / 2 = 1
                                padding=1)
        # 64x64x64 => 64x64x64
        self.conv_4 = nn.Conv1d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                # (1*(64-1) - 64 + 3) / 2 = 1
                                padding=1)
        # 64x64 => 32x64
        self.pool_2 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(32-1) - 64 + 2) / 2 = 0
                                   padding=0)
        # 32x64 => 32x128
        self.conv_5 = nn.Conv1d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                # (1*(32-1) - 32 + 3) / 2 = 1
                                padding=1)
        # 32x128 => 32x128
        self.conv_6 = nn.Conv1d(in_channels=128,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                # (1*(32-1) - 32 + 3) / 2 = 1
                                padding=1)
        # 32x128 => 16x128
        self.pool_3 = nn.MaxPool1d(kernel_size=2,
                                   stride=2,
                                   # (2*(16-1) - 32 + 2) / 2 = 0
                                   padding=0)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(.5)
        self.dense = nn.Linear(16 * 128, self.num_classes)

    def forward(self, x, *args, **kwargs):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = F.relu(x)
        x = self.pool_2(x)
        x = self.conv_5(x)
        x = F.relu(x)
        x = self.conv_6(x)
        x = F.relu(x)
        x = self.pool_3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


# ****** LSTMResClassifier ******

# @deprecated(reason="This model is deprecated. Please use MSRLSTMClassifier instead.")
class LSTMResClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 5
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=128)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=512)
        self.lstm3 = nn.LSTM(input_size=512, hidden_size=512)
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=512)
        self.lstm5 = nn.LSTM(input_size=512, hidden_size=512)
        self.lstm6 = nn.LSTM(input_size=512, hidden_size=128)
        self.fcl1 = nn.Linear(128, 32)
        self.dropout1 = nn.Dropout(.3)
        self.fcl2 = nn.Linear(32, self.num_classes)

    def forward(self, x, *args, **kwargs):
        #  (batch_size, seq_length, feat_dim) to (seq_length, batch_size, feat_dim)
        inp = x.permute(1, 0, 2)
        l1_out, _ = self.lstm1(inp)
        l2_out, _ = self.lstm2(l1_out)
        l3_out, _ = self.lstm3(l2_out)
        l4_out, _ = self.lstm4(l2_out + l3_out)
        l5_out, _ = self.lstm5(l3_out + l4_out)
        l6_out, _ = self.lstm6(l4_out + l5_out)
        out = self.fcl1(l6_out)
        out = self.dropout1(out)
        out = F.relu(out)
        out = self.fcl2(out[-1, :, :])
        return out


# ****** MSRLSTMClassifier ******

class MSRLSTMClassifier(nn.Module):
    """
    for paper Wang, Chenxing, et al. 
    "Combining residual and LSTM recurrent networks for
      transportation mode detection using multimodal sensors 
      integrated in smartphones." IEEE Transactions on Intelligent 
      Transportation Systems 22.9 (2020): 5473-5485.
    """
    def __init__(self):
        super(MSRLSTMClassifier, self).__init__()
        self.num_classes = 5
        
        # CNN参数
        self.cnn_filters = 32
        self.cnn_kernel_size = 3
        
        # ResNet参数
        self.res_filters = [32, 64, 64, 32] 
        self.res_kernel_sizes = [3, 3, 3, 3]
        
        # LSTM参数
        self.lstm_hidden = 128
        self.lstm_dropout = 0.3
        
        # Attention参数
        self.attention_dim = 64
        
        # 卷积层
        self.conv1 = nn.Conv1d(4, self.cnn_filters, self.cnn_kernel_size, padding='same')
        
        # ResNet块
        self.res_conv1 = nn.Conv1d(self.cnn_filters, self.res_filters[0], self.res_kernel_sizes[0], padding='same')
        self.res_conv2 = nn.Conv1d(self.res_filters[0], self.res_filters[1], self.res_kernel_sizes[1], padding='same')
        self.res_shortcut = nn.Conv1d(self.cnn_filters, self.res_filters[1], 1, padding='same')
        self.maxpool = nn.MaxPool1d(2, padding=1)
        
        # LSTM层
        self.lstm = nn.LSTM(self.res_filters[1], self.lstm_hidden, batch_first=True, dropout=self.lstm_dropout)
        
        # Attention层
        self.attention1 = nn.Linear(self.lstm_hidden, self.attention_dim)
        self.attention2 = nn.Linear(self.attention_dim, 1)
        
        # 输出层
        self.fc1 = nn.Linear(self.lstm_hidden, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x, *args, **kwargs):
        # 输入x: (batch_size, seq_length, 4)
        
        # 调整维度顺序以适配CNN
        x = x.permute(0, 2, 1)  # (batch_size, 4, seq_length)
        
        # CNN
        x = F.relu(self.conv1(x))
        
        # ResNet
        identity = x
        x = F.relu(self.res_conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.res_conv2(x))
        identity = self.res_shortcut(identity)
        identity = self.maxpool(identity)
        x = x + identity
        x = F.relu(x)
        
        # 调整维度以适配LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, channels)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention = F.softmax(self.attention2(F.tanh(self.attention1(lstm_out))), dim=1)
        x = torch.sum(attention * lstm_out, dim=1)
        
        # 输出层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ****** DMNClassifier ******

class DMNClassifier(nn.Module):
    """
    DMN (Dual-Modal Network) 用于轨迹片段出行方式分类
    基于Lei等人的方法调整为单任务分类
    """
    
    def __init__(self, feat_dim=4, seq_length=128, embed_dim=128, num_heads=4, 
                 num_layers=2, dim_feedforward=64, num_classes=5, dropout=0.1):
        super(DMNClassifier, self).__init__()
        
        # 超参数设置（默认值）
        self.feat_dim = feat_dim
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.dropout = dropout
        
        # 确保嵌入维度能被注意力头数整除
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        # 1. 特征嵌入模块
        # 空间特征嵌入（经纬度坐标）
        self.spatial_embedding = nn.Conv1d(2, embed_dim, kernel_size=3, padding=1)
        
        # 时间特征嵌入
        # 连续时间特征 (time of day)
        self.temporal_tod_embedding = nn.Conv1d(1, embed_dim, kernel_size=3, padding=1)
        # 离散星期特征 (day of week) - 假设第4个特征是星期
        self.temporal_dow_embedding = nn.Embedding(7, embed_dim)  # 7天
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, embed_dim) * 0.02)
        
        # 2. 多模态编码器
        self.spatial_encoder = self._build_encoder()
        self.temporal_encoder = self._build_encoder()
        
        # 3. 模态间注意力
        self.cross_attention_s2t = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention_t2s = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 4. 前馈网络
        self.spatial_ffn = self._build_ffn()
        self.temporal_ffn = self._build_ffn()
        
        # 5. 全局特征聚合
        self.spatial_attention_pooling = AttentionPooling(embed_dim)
        self.temporal_attention_pooling = AttentionPooling(embed_dim)
        
        # 6. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 2),  # 6e -> 2e 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),      # 2e -> e
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)         # e -> num_classes
        )
        
        # Layer normalization - 需要适配融合后的维度
        self.spatial_norm1 = nn.LayerNorm(embed_dim * 2)  # 融合后是2*embed_dim
        self.spatial_norm2 = nn.LayerNorm(embed_dim)
        self.temporal_norm1 = nn.LayerNorm(embed_dim * 2)  # 融合后是2*embed_dim
        self.temporal_norm2 = nn.LayerNorm(embed_dim)
        
    def _build_encoder(self):
        """构建单个模态的编码器"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward * 4,  # 按论文中的比例
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
    
    def _build_ffn(self):
        """构建前馈网络"""
        return nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.dim_feedforward * 4),  # 2e -> 4*d_ff
            nn.ReLU(),
            nn.Linear(self.dim_feedforward * 4, self.embed_dim * 2),  # 4*d_ff -> 2e
            nn.Dropout(self.dropout)
        )
    
    def forward(self, x, padding_masks=None, feature_masks=None, *args, **kwargs):
        """
        前向传播
        
        Args:
            x: (batch_size, seq_length, feat_dim) 输入特征数据
            padding_masks: (batch_size, seq_length) 填充掩码，1表示有效，0表示填充
            feature_masks: (batch_size, seq_length, feat_dim) 特征掩码，用于预训练
            
        Returns:
            logits: (batch_size, num_classes) 分类logits
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # 提取不同模态的特征
        # 前2个特征：经纬度坐标（空间特征）
        spatial_features = x[:, :, :2]  # (batch_size, seq_len, 2)
        
        # 第3个特征：时间戳（时间特征）
        if feat_dim > 2:
            temporal_features = x[:, :, 2:3]  # (batch_size, seq_len, 1)
        else:
            # 如果没有时间特征，使用序列位置作为代理
            temporal_features = torch.arange(seq_len, device=x.device).float().unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        
        # 星期特征（如果存在第4个特征）
        if feat_dim > 3:
            dow_features = x[:, :, 3].long()  # (batch_size, seq_len)
            dow_features = torch.clamp(dow_features, 0, 6)  # 确保在0-6范围内
        else:
            # 如果没有星期特征，使用默认值
            dow_features = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
        
        # 1. 特征嵌入
        # 空间特征嵌入
        spatial_emb = self.spatial_embedding(spatial_features.transpose(1, 2)).transpose(1, 2)  # (batch_size, seq_len, embed_dim)
        
        # 时间特征嵌入
        temporal_tod_emb = self.temporal_tod_embedding(temporal_features.transpose(1, 2)).transpose(1, 2)  # (batch_size, seq_len, embed_dim)
        temporal_dow_emb = self.temporal_dow_embedding(dow_features)  # (batch_size, seq_len, embed_dim)
        temporal_emb = temporal_tod_emb + temporal_dow_emb
        
        # 添加位置编码
        if seq_len <= self.seq_length:
            spatial_emb = spatial_emb + self.positional_encoding[:, :seq_len, :]
            temporal_emb = temporal_emb + self.positional_encoding[:, :seq_len, :]
        
        # 2. 模态内自注意力编码
        spatial_encoded = self.spatial_encoder(spatial_emb)  # (batch_size, seq_len, embed_dim)
        temporal_encoded = self.temporal_encoder(temporal_emb)  # (batch_size, seq_len, embed_dim)
        
        # 3. 模态间交叉注意力
        # 空间到时间的注意力
        spatial_cross, _ = self.cross_attention_s2t(
            query=temporal_encoded,
            key=spatial_encoded,
            value=spatial_encoded
        )  # (batch_size, seq_len, embed_dim)
        
        # 时间到空间的注意力
        temporal_cross, _ = self.cross_attention_t2s(
            query=spatial_encoded,
            key=temporal_encoded,
            value=temporal_encoded
        )  # (batch_size, seq_len, embed_dim)
        
        # 4. 特征融合
        # 拼接本地和跨模态特征
        spatial_fused = torch.cat([spatial_encoded, spatial_cross], dim=-1)  # (batch_size, seq_len, 2*embed_dim)
        temporal_fused = torch.cat([temporal_encoded, temporal_cross], dim=-1)  # (batch_size, seq_len, 2*embed_dim)
        
        # 前馈网络
        spatial_fused = spatial_fused + self.spatial_ffn(spatial_fused)
        temporal_fused = temporal_fused + self.temporal_ffn(temporal_fused)
        
        # Layer normalization
        spatial_fused = self.spatial_norm1(spatial_fused)
        temporal_fused = self.temporal_norm1(temporal_fused)
        
        # 降维回原始维度
        spatial_final = spatial_fused[:, :, :self.embed_dim]  # 取前embed_dim维
        temporal_final = temporal_fused[:, :, :self.embed_dim]  # 取前embed_dim维
        
        # 5. 全局特征聚合
        # 使用注意力池化
        spatial_global = self.spatial_attention_pooling(spatial_final)  # (batch_size, embed_dim)
        temporal_global = self.temporal_attention_pooling(temporal_final)  # (batch_size, embed_dim)
        
        # 特征连接和交互
        global_concat = torch.cat([spatial_global, temporal_global], dim=-1)  # (batch_size, 2*embed_dim)
        global_interact = spatial_global * temporal_global  # (batch_size, embed_dim)
        
        # 最终全局特征
        global_features = torch.cat([global_concat, global_interact], dim=-1)  # (batch_size, 3*embed_dim)
        
        # 增强特征表示
        enhanced_features = torch.cat([
            global_features,
            torch.abs(spatial_global - temporal_global),  # 差异特征
            spatial_global + temporal_global,  # 和特征
            torch.max(spatial_global, temporal_global)   # 最大特征
        ], dim=-1)  # (batch_size, 6*embed_dim)
        
        # 6. 分类
        logits = self.classifier(enhanced_features)  # (batch_size, num_classes)
        
        return logits


class AttentionPooling(nn.Module):
    """注意力池化模块"""
    
    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            pooled: (batch_size, embed_dim)
        """
        # 计算注意力权重
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # 加权聚合
        pooled = torch.sum(attn_weights * x, dim=1)  # (batch_size, embed_dim)
        
        return pooled