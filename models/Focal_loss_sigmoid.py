import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.EPS = 1e-12

    def forward(self, input, target):
        pt = input.view(-1) * (target.view(-1) == 1.).float() + (
            1 - input.view(-1)) * (target.view(-1) == 0.).float()
        loss=-self.alpha*(torch.pow((1-pt),self.gamma))*torch.log(pt+self.EPS)*(target.view(-1)==1.).float()-\
        (1 - self.alpha)* (torch.pow((1 - pt), self.gamma)) * torch.log(pt + self.EPS) * (target.view(-1) == 0.).float()
        self.loss = loss.mean()
        return self.loss
