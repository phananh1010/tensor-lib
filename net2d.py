import torch as tc
import numpy as np

class ConvBlock(tc.nn.Module):
    #TODO: conv2d->conv2d block for compression? and feature representation
    def __init__(self, in_channels, out_channels, residual=True, compress_factor=2):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.conv2d = tc.nn.Sequential(
            #structure below has been used by yeon et al for superres
            tc.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #structure below is for inpainting only
            #tc.nn.Conv2d(in_channels, out_channels//compress_factor, kernel_size=1, stride=1, padding=0),
            #tc.nn.Conv2d(out_channels//compress_factor, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            tc.nn.BatchNorm2d(out_channels),
            tc.nn.LeakyReLU(0.2, inplace=True),        
        )
    
    def forward(self, x):
        if self.residual==True:
            x = x + self.conv2d(x)
        else:
            x = self.conv2d(x)
        return x
    
#############decoder block#############
    
class Decoder(tc.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Decoder, self).__init__()
        self.deconv = tc.nn.Sequential(
            tc.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            tc.nn.LeakyReLU(0.2, inplace=True),
            tc.nn.BatchNorm2d(output_channels),
        )
    def forward(self, x):
        x = tc.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.deconv(x)
        return x
################################################
#############encoder block#############
class Encoder(tc.nn.Module):
    def __init__(self, input_channels, hidden, output_channels):
        super(Encoder, self).__init__()
        self.conv1 = tc.nn.Sequential(
            #t.nn.Conv2d(3, hidden, kernel_size=3, stride=1, padding=2, dilation=2),
            tc.nn.Conv2d(input_channels, hidden, kernel_size=3, stride=1, padding=1),
            tc.nn.BatchNorm2d(hidden),
            tc.nn.LeakyReLU(0.2, inplace=True),        
        )
        
        self.conv2 = tc.nn.Sequential(
            tc.nn.Conv2d(hidden, output_channels, kernel_size=3, stride=2, padding=1),
            tc.nn.BatchNorm2d(output_channels),
            tc.nn.LeakyReLU(0.2, inplace=True),        
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x 
    
##########IMAGER: create image output from conv input###############

class Imager(tc.nn.Module):
    #TODO: output image from decoded tensor
    def __init__(self, input_channel):
        super(Imager, self).__init__()
        self.conv = tc.nn.Conv2d(input_channel, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        #x = tc.tanh(x)
        return x

##########flattener: use aapool to flatten data along channel axis, and use a fc layer to produce output
class Flatenner(tc.nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Flatenner, self).__init__()
        self.aapool = tc.nn.AdaptiveAvgPool2d(1)#compress filter maps (b, c, h, w) to (b, c)
        self.fc = tc.nn.Linear(input_channel, output_channel)
        
    def forward(self, x):
        x = self.aapool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x