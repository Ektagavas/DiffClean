import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T
from models.SSR_Net_model.SSR_Net_model import SSRNet

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
        


    def forward(self, inputs, target, size_average=True):

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

        if size_average:
            return loss.mean()
        return loss.sum()


class AgeLoss(torch.nn.Module):
    def __init__(self, device):
        super(AgeLoss, self).__init__()
        print('Loading SSR-Net')
        self.age_model = SSRNet()
        model_file = 'checkpoint/ssrnet_finetuned.pth'
        loaded_model = torch.load(model_file)
        self.age_model.load_state_dict(loaded_model['state_dict'])
        self.age_model = self.age_model.to(device)
        self.age_model.eval()

        self.device = device

        self.transform = T.Compose([
            T.Lambda(lambda t: t * 0.5 + 0.5),  # Undo Normalize
            T.Resize((64, 64), interpolation=T.InterpolationMode.BILINEAR),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.loss = AdjustSmoothL1Loss()

    def process_batch(self,batch_tensor):
        return torch.stack([
            self.transform(img) for img in batch_tensor
        ])

    def inference(self, img):
        img = self.process_batch(img)
        with torch.set_grad_enabled(False):
            results = self.age_model(img)
        return results

    def forward(self, img1, age2):
        age1 = self.inference(img1)
        return self.loss(age1, age2)

