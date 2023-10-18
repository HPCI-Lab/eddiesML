import numpy as np
import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, DEVICE):
        super(WeightedCrossEntropyLoss, self).__init__()
        
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights = class_weights.to(DEVICE)
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets)

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
