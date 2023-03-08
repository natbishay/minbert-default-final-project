import torch.nn as nn
import torch

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        mask = target == 9
        high_cost = (loss * mask.float()).mean()
        return loss + high_cost