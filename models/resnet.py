from torchvision import models
import torch.nn as nn

class ResBase(nn.Module):
    def __init__(self, option='resnet50', pretrained=True):
        super(ResBase, self).__init__()
        self.output_dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pretrained)
            self.output_dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pretrained)
            self.output_dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pretrained)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pretrained)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pretrained)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.output_dim)
        return x

