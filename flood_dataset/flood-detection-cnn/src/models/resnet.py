from torch import nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def freeze_layers(self, num_layers_to_freeze=5):
        for param in list(self.resnet.parameters())[:-num_layers_to_freeze]:
            param.requires_grad = False

    def unfreeze_layers(self, num_layers_to_unfreeze=5):
        for param in list(self.resnet.parameters())[-num_layers_to_unfreeze:]:
            param.requires_grad = True