import torch
import torch.nn as nn 
from torchvision import models
from torchsummary import summary
from efficientnet_pytorch import EfficientNet

# 1 x 96 x 96
"""
class SiameseEfficientNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block = self.loading_efficient_net()

        self.fc = nn.Sequential(
            nn.Linear(4608, 512),
            #nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

        self.out = nn.Linear(256, 1)


    def forward_one_side(self, x):
        x = self.cnn_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x):
        #out1 = self.forward_one_side(in1)
        #out2 = self.forward_one_side(in2)

        #dis = torch.abs(out1 - out2)
        #out = self.out(dis)
        return self.cnn_block(x)
"""

def loading_efficient_net():
    eff_model = EfficientNet.from_pretrained('efficientnet-b4')
    for param in eff_model.parameters():
        param.requires_grad = False

    eff_model = nn.Sequential(*list(eff_model.children())[:-4])
    return eff_model


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.normal_(m.bias.data, mean=0.5, std=0.01)

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.2)
        if m.bias is not None:
            nn.init.normal_(m.bias.data, mean=0.0, std=0.01)


class SEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block = loading_efficient_net()


    def forward(self, x):
        x = self.cnn_block(x)
        return x 

if __name__ == "__main__":
    eff_model = SEN().cuda()
    x = torch.randn((128, 3, 224, 224)).cuda()

    y = eff_model(x)

    #summary(eff_model, x.shape)