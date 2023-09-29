import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        return ce_loss(logits, targets)

