import numpy as np
import torch


def mean_columnwise_log_loss(y_true, y_pred):
    # Clip the predictions to prevent log(0)
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    # Compute log loss for each label (column)
    log_loss_per_column = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=0)

    # Return the mean log loss across all labels
    return np.mean(log_loss_per_column)


# Pytorch Mean Columnwise Log Loss
def mean_columnwise_log_loss_torch(y_true, y_pred):
    # Clip the predictions to prevent log(0)
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    # Compute log loss for each label (column)
    log_loss_per_column = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean(dim=0)

    # Return the mean log loss across all labels
    return log_loss_per_column.mean()
