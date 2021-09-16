import torch
import torch.nn as nn 
import torch.nn.functional as F 
# 1 x 96 x 96

class SiameseNetwork_Kaggle(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 16, 48, 48

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 32, 24, 24

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 32, 12, 12
        )


        self.fc = nn.Sequential(
            nn.Linear(32*12*12, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 128)
        )


    def forward_one_side(self, x):
        x = self.cnn_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, in1, in2):
        out1 = self.forward_one_side(in1)
        out2 = self.forward_one_side(in2)
        return out1, out2 


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
    net = SiameseNetwork()

    in1 = torch.randn((16, 1, 96, 96))
    in2 = in1
    out = net(in1, in2)
    print(out.shape)