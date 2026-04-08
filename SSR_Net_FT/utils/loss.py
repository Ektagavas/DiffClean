import torch
from torch import nn


age_groups = [(0,2), (3,6), (7,9), (10,14), (15,19), (20,29), (30,39), (40,49), (50,69), (70,float('inf'))]
def get_age_group(age):
    for index, (start, end) in enumerate(age_groups):
        if start <= age <= end:
            return index

class AdjustSmoothL1Loss(nn.Module):

    def __init__(self, num_features=1, momentum=0.1, beta=1. /9):
        super(AdjustSmoothL1Loss, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.beta = beta
        self.register_buffer(
            'running_mean', torch.empty(num_features).fill_(beta)
        )
        self.register_buffer('running_var', torch.zeros(num_features))
        
        self.age_group_weights = {
    "(0;2)": 1.0, "(3;6)": 1.0, "(7;9)": 1.0, "(10;14)":3.0, "(15;19)": 3.0,
    "(20;29)": 3.0, "(30;39)": 1.0, "(40;49)": 1.0, "(50;69)": 1.0}
        self.total = 0
        for key in self.age_group_weights:
            self.total += self.age_group_weights[key]


    def forward(self, inputs, target, age_groups, size_average=True):

        n = torch.abs(inputs -target)
        with torch.no_grad():
            if torch.isnan(n.var(dim=0)).sum().item() == 0:
                self.running_mean = self.running_mean.to(n.device)
                self.running_mean *= (1 - self.momentum)
                self.running_mean += (self.momentum * n.mean(dim=0))
                self.running_var = self.running_var.to(n.device)
                self.running_var *= (1 - self.momentum)
                self.running_var += (self.momentum * n.var(dim=0))


        beta = (self.running_mean - self.running_var)
        beta = beta.clamp(max=self.beta, min=1e-3)

        beta = beta.view(-1, self.num_features).to(n.device)
        n = n.view((-1,1))
        
        cond = n < beta.expand_as(n)
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        normalized_weights = torch.tensor([self.age_group_weights[grp]/self.total for grp in age_groups], dtype=torch.float32, device=n.device)
        weighted_loss = loss * normalized_weights

        if size_average:
            return weighted_loss.mean()
        return weighted_loss.sum()
