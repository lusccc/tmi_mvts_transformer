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
            return TSTransformerEncoderClassifier(**utils.load_model_hyperparams(config['feature_branch_hyperparams']))
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

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassifier(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassifier, self).__init__()

        self.hyperparams = locals()
        
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        # 使用更高效的特征提取方式而不是简单展平
        # 利用全局平均池化和全局最大池化捕获序列中的关键信息
        # 这大大减少了参数量，同时保留了时间序列中的关键特征
        
        class TemporalFeatureExtractor(nn.Module):
            def __init__(self, d_model, max_len, num_classes):
                super(TemporalFeatureExtractor, self).__init__()
                # 输入: (batch_size, seq_length * d_model) 来自展平的序列
                # 也可以接收 (batch_size, seq_length, d_model) 的输入并处理
                
                self.d_model = d_model
                self.max_len = max_len
                
                # 特征提取层
                self.feature_extractor = nn.Sequential(
                    # 关注时间维度的卷积，保留全部特征维度
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.BatchNorm1d(d_model),
                    nn.GELU(),
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.BatchNorm1d(d_model),
                    nn.GELU(),
                )
                
                # 多尺度聚合模块
                hidden_size = d_model * 2  # 平均池化和最大池化的拼接
                
                # 分类头
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, num_classes)
                )
            
            def forward(self, x):
                # 输入可能是展平的 (batch_size, seq_length * d_model)
                if x.dim() == 2:
                    batch_size = x.size(0)
                    x = x.view(batch_size, self.max_len, self.d_model)
                
                # 转置为卷积1D格式 (batch_size, d_model, seq_length)
                x = x.transpose(1, 2)
                
                # 应用特征提取
                x = self.feature_extractor(x)
                
                # 全局池化: 两种聚合方式捕获不同类型的特征
                avg_pool = torch.mean(x, dim=2)  # 全局平均池化
                max_pool, _ = torch.max(x, dim=2)  # 全局最大池化
                
                # 拼接不同的池化结果
                x = torch.cat([avg_pool, max_pool], dim=1)
                
                # 分类
                x = self.classifier(x)
                return x
        
        return TemporalFeatureExtractor(d_model, max_len, num_classes)

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings, i.e., make padding to 0s
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output


# 新增一个专用于双分支模型的TSTransformerEncoder子类
class TSTransformerEncoderForDualBranch(TSTransformerEncoder):
    """为双分支模型特别优化的TSTransformerEncoder版本
    
    与标准TSTransformerEncoder的主要区别:
    1. 输出层被替换为Flatten
    2. 优化了内部结构以支持DualTSTransformerEncoderClassifier
    """
    
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderForDualBranch, self).__init__(
            feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, 
            dropout, pos_encoding, activation, norm, freeze
        )
        # 替换输出层为Flatten而不是线性层
        self.output_layer = nn.Flatten()
        
    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length * d_model) 扁平化的特征向量
        """
        # 重用父类的Transformer编码器逻辑
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)
        
        # 应用掩码并扁平化
        output = output * padding_masks.unsqueeze(-1)  # 对填充位置置零
        output = self.output_layer(output)  # 扁平化
        
        return output


class DualTSTransformerEncoderClassifier(nn.Module):
    def __init__(self, trajectory_branch_hyperparams, feature_branch_hyperparams, num_classes, dropout=0.1,
                 activation='gelu'):
        super(DualTSTransformerEncoderClassifier, self).__init__()
        self.num_classes = num_classes
        
        # 记录原始特征维度和序列长度，用于后续处理
        self.trajectory_feat_dim = trajectory_branch_hyperparams.get('feat_dim')
        self.feature_feat_dim = feature_branch_hyperparams.get('feat_dim')
        self.trajectory_max_len = trajectory_branch_hyperparams.get('max_len')
        self.feature_max_len = feature_branch_hyperparams.get('max_len')
        
        # 创建分支模型，使用专门为双分支设计的TSTransformerEncoder
        self.trajectory_branch = TSTransformerEncoderForDualBranch(**trajectory_branch_hyperparams)
        self.feature_branch = TSTransformerEncoderForDualBranch(**feature_branch_hyperparams)

        # 保存d_model，用于维度调整
        self.trajectory_d_model = self.trajectory_branch.d_model
        self.feature_d_model = self.feature_branch.d_model
        
        # 轨迹分支特征提取器 - 与单分支保持一致
        self.encoder1_output_layer = self.build_feature_extractor(self.trajectory_d_model)
        
        # 特征分支特征提取器 - 与单分支保持一致
        self.encoder2_output_layer = self.build_feature_extractor(self.feature_d_model)

        # 分类头 - 注意输入维度的变化
        combined_feat_size = (self.trajectory_d_model * 2) + (self.feature_d_model * 2)  # 两个分支的拼接特征
        self.output_layer = nn.Sequential(
            nn.Linear(combined_feat_size, combined_feat_size // 2),
            nn.BatchNorm1d(combined_feat_size // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(combined_feat_size // 2, num_classes)
        )

        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)

    def build_feature_extractor(self, d_model):
        """构建与单分支一致的特征提取器"""
        return nn.Sequential(
            # 特征提取层 - 保持特征维度不变
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
            # 注意这里不包含池化操作，池化在forward中进行以便获取两种池化结果
        )
        
    def forward(self, X1, padding_mask1, X2, padding_mask2):
        """双分支转换器分类器的前向传播
        
        Args:
            X1: 轨迹分支输入，形状 (batch_size, seq_length, feat_dim)
            padding_mask1: 轨迹分支填充掩码，形状 (batch_size, seq_length)
            X2: 特征分支输入，形状 (batch_size, seq_length, feat_dim)
            padding_mask2: 特征分支填充掩码，形状 (batch_size, seq_length)
            
        Returns:
            output: 分类结果，形状 (batch_size, num_classes)
        """
        # 轨迹分支 - 获取扁平化特征
        output1 = self.trajectory_branch(X1, padding_mask1)  # (batch_size, seq_length * d_model)
        
        # 调整维度为卷积1D格式
        batch_size = output1.size(0)
        output1 = output1.view(batch_size, self.trajectory_max_len, self.trajectory_d_model)
        output1 = output1.transpose(1, 2)  # (batch_size, d_model, seq_length)
        
        # 应用特征提取
        output1 = self.encoder1_output_layer(output1)
        
        # 应用双池化策略 - 与单分支一致
        avg_pool1 = torch.mean(output1, dim=2)  # 全局平均池化
        max_pool1, _ = torch.max(output1, dim=2)  # 全局最大池化
        output1 = torch.cat([avg_pool1, max_pool1], dim=1)  # (batch_size, d_model*2)
        
        # 特征分支 - 获取扁平化特征
        output2 = self.feature_branch(X2, padding_mask2)  # (batch_size, seq_length * d_model)
        
        # 调整维度为卷积1D格式
        output2 = output2.view(batch_size, self.feature_max_len, self.feature_d_model)
        output2 = output2.transpose(1, 2)  # (batch_size, d_model, seq_length)
        
        # 应用特征提取
        output2 = self.encoder2_output_layer(output2)
        
        # 应用双池化策略 - 与单分支一致
        avg_pool2 = torch.mean(output2, dim=2)  # 全局平均池化
        max_pool2, _ = torch.max(output2, dim=2)  # 全局最大池化
        output2 = torch.cat([avg_pool2, max_pool2], dim=1)  # (batch_size, d_model*2)
        
        # 融合两个分支的特征
        output = torch.cat([output1, output2], dim=1)  # (batch_size, d_model1*2 + d_model2*2)
        
        # 分类
        output = self.output_layer(output)
        
        return output


class LambdaLayer(nn.Module):
    def __init__(self):
        super(LambdaLayer, self).__init__()

    def forward(self, x):
        return x


# ****** below are COMPARATIVE MODELS:

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