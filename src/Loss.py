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
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)
        dice_scores = torch.zeros(num_classes, dtype=logits.dtype, device=logits.device)

        for class_idx in range(num_classes):
            class_probs = probs[:, class_idx, ...]
            class_labels = (labels == class_idx).float()

            intersection = (class_probs * class_labels).sum()
            union = class_probs.sum() + class_labels.sum()

            dice_scores[class_idx] = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1.0 - dice_scores.mean()
        return loss
