import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from tmi.utils import utils


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
        return DualTSTransformerEncoderClassifier(utils.load_model_hyperparams(config['trajectory_branch_hyperparams']),
                                                  utils.load_model_hyperparams(config['feature_branch_hyperparams']),
                                                  len(data.feature_data.class_names), dropout=config['dropout'],
                                                  activation=config['activation'], emb1_size=config['emb_size'],
                                                  emb2_size=config['emb_size'])
    if task == 'trajectory_branch_classification':
        return TSTransformerEncoderClassifier(**utils.load_model_hyperparams(config['trajectory_branch_hyperparams']),
                                              num_classes=len(data.class_names))
    if task == 'feature_branch_classification':
        return TSTransformerEncoderClassifier(**utils.load_model_hyperparams(config['feature_branch_hyperparams']),
                                              num_classes=len(data.class_names))
    if task in ['feature_branch_classification_from_scratch', 'trajectory_branch_classification_from_scratch']:
        feat_dim = data.noise_feature_df.shape[1]
        max_seq_len = data.max_seq_len
        return TSTransformerEncoderClassifier(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                              config['num_layers'], config['dim_feedforward'],
                                              num_classes=len(data.class_names), dropout=config['dropout'],
                                              pos_encoding=config['pos_encoding'], activation=config['activation'],
                                              norm=config['normalization_layer'], freeze=config['freeze'])
    if 'cnn_classification' in task:
        return CNN1DClassifier_128()
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
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

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


class DualTSTransformerEncoderClassifier(nn.Module):
    def __init__(self, trajectory_branch_hyperparams, feature_branch_hyperparams, num_classes, dropout=0.1,
                 activation='gelu', emb1_size=64, emb2_size=64):
        super(DualTSTransformerEncoderClassifier, self).__init__()
        self.num_classes = num_classes
        self.trajectory_branch = TSTransformerEncoder(**trajectory_branch_hyperparams)
        self.feature_branch = TSTransformerEncoder(**feature_branch_hyperparams)

        # replace output layer in original model
        self.trajectory_branch.output_layer = nn.Flatten()  # i.e., reshape()
        self.feature_branch.output_layer = nn.Flatten()
        # then, create new output layer
        self.encoder1_output_layer = nn.Linear(self.trajectory_branch.d_model * self.trajectory_branch.max_len,
                                               emb1_size)
        self.encoder2_output_layer = nn.Linear(self.feature_branch.d_model * self.feature_branch.max_len, emb2_size)

        # create output layer for this DualTSTransformerEncoderClassifier
        # self.output_layer = nn.Linear(emb1_size + emb2_size, num_classes)

        # the emb from each branch will be added together, so here make output emb size same as each branch
        self.output_layer = nn.Linear(emb1_size, num_classes)

        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, X1, padding_mask1, X2, padding_mask2):
        output1 = self.trajectory_branch(X1, padding_mask1)
        output1 = self.encoder1_output_layer(output1)  # (batch_size, emb1_size)
        output2 = self.feature_branch(X2, padding_mask2)
        output2 = self.encoder2_output_layer(output2)  # (batch_size, emb2_size)
        # output = torch.cat([output1, output2], dim=1)  # (batch_size, emb1_size+emb2_size
        output = torch.add(output1, output2)  # (batch_size, emb1_size+emb2_size
        # """add gelu:"""
        # output = self.act(output)

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