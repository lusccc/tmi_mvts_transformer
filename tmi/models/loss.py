import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):
    task = config['task']
    if task in ['denoising_imputation_pretrain', 'imputation_pretrain', 'denoising_pretrain']:
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element
        # return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

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

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

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

        """
        below deprecated
        """
        # all_loss = []
        # for m, y_p, y_t in zip(mask, y_pred, y_true):
        #     m_col = m[:,0]
        #     n_non_mask = torch.count_nonzero(m_col) # number of 1s
        #     n_mask = m_col.shape[0] - n_non_mask
        #
        #     mask_ratio = n_mask/m_col.shape[0]
        #     if .15 < mask_ratio < .3:
        #         loss = self.mse_loss(torch.masked_select(y_p, m), torch.masked_select(y_t, m))
        #     else:
        #         loss = self.mse_loss(y_p, y_t)
        #
        #     all_loss.append(loss)
        #
        # return torch.cat(all_loss)

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
