import torchvision.models as models
import torch.nn as nn
import torch
from torchvision.models.inception import Inception3

# class InceptionV3(nn.Module):
#     def __init__(self, num_classes, aux_logits=True, transform_input=False):
#         super(InceptionV3, self).__init__()
#         model = Inception3(num_classes=num_classes, aux_logits=aux_logits,transform_input=transform_input)
#         self.model = model
#
#     def forward(self, x):
#         x = self.model(x)
#         return x

class InceptionV3(nn.Module):
    def __init__(self, num_classes, aux_logits=False, transform_input=False):
        super(InceptionV3, self).__init__()
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs,num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
#        model = Inception3(num_classes=num_classes, aux_logits=aux_logits,transform_input=transform_input)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x
