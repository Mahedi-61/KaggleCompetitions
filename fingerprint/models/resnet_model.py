import torch
import torch.nn as nn 
from torchvision import models
from torchsummary import summary
# 1 x 96 x 96

class SiameseResentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_model = self.loading_resnet18()
        self.cnn_block = nn.Sequential(
            nn.Sequential(*list(self.res_model.children())[:-2]),
        )

        self.fc = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

        self.out = nn.Linear(256, 1)

    def loading_resnet18(self):
        res_model = models.resnet18(pretrained=True)
        for param in res_model.parameters():
            param.requires_grad = False

        return res_model

    def forward_one_side(self, x):
        x = self.cnn_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, in1, in2):
        out1 = self.forward_one_side(in1)
        out2 = self.forward_one_side(in2)

        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.normal_(m.bias.data, mean=0.5, std=0.01)

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.2)
        if m.bias is not None:
            nn.init.normal_(m.bias.data, mean=0.0, std=0.01)


if __name__ == "__main__":
    net = SiameseResentNetwork().to("cuda")