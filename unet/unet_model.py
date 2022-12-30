""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8, groups=[1, 8])
        self.maxpool = Down(1, 8)
        self.down1 = DoubleConv(8, 16, groups=[8, 16])
        self.down2 = DoubleConv(16, 32, groups=[16, 32])
        self.down3 = DoubleConv(32, 64, groups=[32, 64])
        factor = 2 if bilinear else 1
        #factor = 1
        self.down4 = DoubleConv(64, 128 // factor, groups=[64, 128])
        self.up1 = Up(128, 64 // factor, bilinear, groups=[64, 64])
        self.up2 = Up(64, 32 // factor, bilinear, groups=[32, 32])
        self.up3 = Up(32, 16 // factor, bilinear, groups=[16, 16])
        self.up4 = Up(16, 8, bilinear, groups=[8, 8])
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        #x_np = x.cpu().numpy()
        x1 = self.inc(x)
        #x1_np = x1.cpu().detach().numpy()
        
        x2 = self.maxpool(x1)
        x2 = self.down1(x2)
        #x2_np = x2.cpu().detach().numpy()
        
        x3 = self.maxpool(x2)
        x3 = self.down2(x3)
        #x3_np = x3.cpu().detach().numpy()
        
        x4 = self.maxpool(x3)
        x4 = self.down3(x4)
        #x4_np = x4.cpu().detach().numpy()
        
        x5 = self.maxpool(x4)
        x5 = self.down4(x5)
        #x5_np = x5.cpu().detach().numpy()
        
        x = self.up1(x5, x4)
        #x_np = x.cpu().detach().numpy()
        
        x = self.up2(x, x3)
        #x_np = x.cpu().detach().numpy()
        
        x = self.up3(x, x2)
        #x_np = x.cpu().detach().numpy()
        
        x = self.up4(x, x1)
        #x_np = x.cpu().detach().numpy()
        
        logits = self.outc(x)
        #l_np = logits.cpu().detach().numpy()
        return logits
