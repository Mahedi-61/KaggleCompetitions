import torch 
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.main(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.main(x)
    
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        mid_channels = in_channels // 2
        self.up_conv =  nn.ConvTranspose2d(in_channels, mid_channels, 
                               kernel_size=2, stride=2, padding=0)
        
        self.double_conv =  DoubleConv(mid_channels*2, out_channels)

        
    def forward(self, x, copy):
        x = self.up_conv(x)
        #pad_lower = (copy.size()[2] - x.size()[2]) // 2
        #pad_upper = copy.size()[2] - pad_lower
        #copy = copy[:, :, pad_lower:pad_upper, pad_lower:pad_upper]
        x = torch.cat([copy, x], dim=1)
        
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, img_channels, num_classes, features=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        
        self.dc1 = DoubleConv(img_channels, features)
        self.dc2 = DoubleConv(features, features*2)
        self.dc3 = DoubleConv(features*2, features*4)
        self.dc4 = DoubleConv(features*4, features*8)
        self.dc5 = DoubleConv(features*8, features*16)
        
        self.up1 = Up(features*16, features*8)
        self.up2 = Up(features*8, features*4)
        self.up3 = Up(features*4, features*2)
        self.up4 = Up(features*2, features)
        
        self.final = nn.Conv2d(features, num_classes, 1, 1, 0)
        
    def forward(self, x):
        #contracting path
        d1 = self.dc1(x)
        d2 = self.dc2(self.max_pool(d1))
        d3 = self.dc3(self.max_pool(d2))
        d4 = self.dc4(self.max_pool(d3))
        x = self.dc5(self.max_pool(d4)) #bottlenek
        
        #expansive path
        x = self.up1(x, d4)
        x = self.up2(x, d3)
        x = self.up3(x, d2)
        x = self.up4(x, d1)
        return self.final(x)
    
