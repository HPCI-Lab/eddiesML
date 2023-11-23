import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        return ce_loss(logits, targets)
    
class SoftDiceLoss(nn.Module):
    def __init__(self, class_weights, smooth=1):
        super(SoftDiceLoss, self).__init__()
        
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, logits, labels):
        num_classes = logits.size(1)
        dice_scores = torch.zeros(num_classes, dtype=logits.dtype, device=logits.device)

        for class_idx in range(num_classes):
            class_probs = logits[:, class_idx]
            class_labels = (labels == class_idx).float()

            intersection = (class_probs * class_labels).sum()
            union = class_probs.sum() + class_labels.sum()

            dice_scores[class_idx] = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        weighted_mean = (self.class_weights[0]*dice_scores[0] + self.class_weights[1]*dice_scores[1] + self.class_weights[2]*dice_scores[2])
        loss = 1.0 - weighted_mean
        return loss

# Focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, class_weights=[0.0238, 0.5028, 0.4734], reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, input, target):
        num_classes = input.size(1)
        probs = F.softmax(input, dim=1)
        one_hot = F.one_hot(target, num_classes=input.size(1)).float()
        pt = (probs * one_hot).sum(1) + 1e-9
        focal_loss = -((1 - pt).pow(self.gamma)) * torch.log(pt)
        
        if self.alpha is not None:
            alpha_factor = one_hot * self.alpha + (1 - one_hot) * (1 - self.alpha)
            focal_loss = focal_loss * alpha_factor

        if self.class_weights is not None:
            class_weights = torch.tensor(self.class_weights, dtype=torch.float, device=input.device)
            # Expand class_weights to match the size of focal_loss
            #class_weights = class_weights.view(1, -1)
            class_weights = (self.class_weights[0]*focal_loss[0] + self.class_weights[1]*focal_loss[1] + self.class_weights[2]*focal_loss[2])
            focal_loss = focal_loss * class_weights

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")

    
#Weighted Tversky loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, class_weights=None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        assert y_pred.size(0) == y_true.size(0), "Batch size mismatch between y_pred and y_true"

        y_pred = torch.sigmoid(y_pred)

        # Initialize losses for each class
        tversky_losses = torch.zeros(3, dtype=y_pred.dtype, device=y_pred.device)

        for class_idx in range(3):
            class_probs = y_pred[:, class_idx]
            class_labels = (y_true == class_idx).float()

            # Apply class weights
            if self.class_weights is not None:
                class_weights = self.class_weights[class_idx]
                class_probs = class_probs * class_weights
                class_labels = class_labels * class_weights

            # Tversky loss
            intersection = (class_probs * class_labels).sum()
            tversky = (intersection + self.smooth) / ((class_probs + class_labels).sum() - intersection + self.smooth)
            tversky_losses[class_idx] = 1 - tversky

        # Average Tversky losses across classes
        loss = tversky_losses.mean()

        return loss
# Loss - combination of dice loss and tversky loss to address class imbalance and multi scale nature of eddies

class TverskyDiceLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, class_weights=None):
        super(TverskyDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        assert y_pred.size(0) == y_true.size(0), "Batch size mismatch between y_pred and y_true"

        y_pred = torch.sigmoid(y_pred)

        # Initialize losses for each class
        tversky_losses = torch.zeros(3, dtype=y_pred.dtype, device=y_pred.device)
        dice_losses = torch.zeros(3, dtype=y_pred.dtype, device=y_pred.device)

        for class_idx in range(3):
            class_probs = y_pred[:, class_idx]
            class_labels = (y_true == class_idx).float()

            # Apply class weights
            if self.class_weights is not None:
                class_weights = self.class_weights[class_idx]
                class_probs = class_probs * class_weights
                class_labels = class_labels * class_weights

            # Tversky loss
            intersection = (class_probs * class_labels).sum()
            tversky = (intersection + self.smooth) / ((class_probs + class_labels).sum() - intersection + self.smooth)
            tversky_losses[class_idx] = 1 - tversky

            # Dice loss
            dice_loss = 1 - (2 * intersection + self.smooth) / (class_probs.sum() + class_labels.sum() + self.smooth)
            dice_losses[class_idx] = dice_loss

        # Combine Tversky and Dice losses
        combined_loss = tversky_losses.mean() + dice_losses.mean()

        return combined_loss
    
