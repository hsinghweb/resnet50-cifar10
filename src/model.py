import torch
import torch.nn as nn
import torchvision.models as models

class CIFAR10ResNet50(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(CIFAR10ResNet50, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer to handle CIFAR-10's 32x32 images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool as we have smaller images
        
        # Modify final fc layer for CIFAR-10 classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)