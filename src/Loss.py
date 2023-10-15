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
