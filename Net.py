import torch
import torch.nn as nn
import torch.nn.functional as F



class FTU(nn.Module): #Takes in Tensor of Dimension B,6,W,H
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6,out_channels=32,kernel_size=7,dilation=1,padding="same")
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,dilation=1,padding="same")
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,dilation=1,padding="same")
    
    def forward(self,pair):
        x = pair
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        return x
        

class ConfMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=12,out_channels=128,kernel_size=7,dilation=1,padding="same")
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=5,dilation=1,padding="same")
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,dilation=1,padding="same")
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,dilation=1,padding="same")
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=7,dilation=1,padding="same")
        self.conv6 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,dilation=1,padding="same")
        self.conv7 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,dilation=1,padding="same")
        self.conv8 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,dilation=1,padding="same")
    
    def forward(self,combined): # Takes in Tensor of dimension B,12,W,H
        x = combined
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.sigmoid(self.conv8(x))
        out1,out2,out3 = torch.split(x,[1,1,1],dim=1)
        return out1,out2,out3

class Generator(nn.Module): # Takes in 4 Tenors(1 of shape B,12,W,H and 3 of shape B,6,W,H)
    def __init__(self):
        super().__init__()
        self.ftu1 = FTU()
        self.ftu2 = FTU()
        self.ftu3 = FTU()
        self.cm = ConfMap()

    def forward(self,combined,pair1,pair2,pair3):
        x1 = self.ftu1(pair1)
        x2 = self.ftu2(pair2)
        x3 = self.ftu3(pair3)
        x1cm,x2cm,x3cm = self.cm(combined)
        output = torch.mul(x1,x1cm) + torch.mul(x2,x2cm) + torch.mul(x3,x3cm)

        return output

'''Credit for the Discriminator Model:
        GitHub:xahidbuffon/FUnIE-GAN'''
class Disciminator(nn.Module):  #Takes raw image + ground truth or output of generator 
    def __init__(self,in_channels=3):
        super().__init__()

        def block(in_filters,out_filters,bn=True):
            layers = [nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=4,stride=2,padding=1)]
            if(bn):
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(.2,inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(in_channels*2,32,bn=False),
            *block(32,64),
            *block(64,128),
            *block(128,256),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(256,1,4,padding=1,bias=False)
        )

    def forward(self,img1,img2):
        img_input = torch.cat((img1,img2),dim=1)
        return self.model(img_input)