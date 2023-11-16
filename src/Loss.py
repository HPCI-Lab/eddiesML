import torch
import numpy as np
import torch.nn as nn

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