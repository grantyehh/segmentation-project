import torch
import torchvision
from torch import nn
from torch.nn import functional as F
# Semantic Model (Deeplab V3)

class ASSP(nn.Module):
  def __init__(self,in_channels,out_channels = 256):
    super(ASSP,self).__init__()
    
    
    self.relu = nn.ReLU(inplace=True)

    self.conv0 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0,
                          bias=False)
    
    self.bn0 = nn.BatchNorm2d(out_channels)  

    self.conv1 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 2,
                          dilation = 2,
                          bias=False)
        
    self.bn1 = nn.BatchNorm2d(out_channels)
    
    self.conv2 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 3,
                          dilation = 3,
                          bias=False)
    
    self.bn2 = nn.BatchNorm2d(out_channels)
    
    self.conv3 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 4,
                          dilation = 4,
                          bias=False)
    
    self.bn3 = nn.BatchNorm2d(out_channels)
    
    self.conv4 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 5,
                          dilation = 5,
                          bias=False)
    
    self.bn4 = nn.BatchNorm2d(out_channels)
    
    self.conv5 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0,
                          bias=False)
    
    self.bn5 = nn.BatchNorm2d(out_channels)



    self.convf = nn.Conv2d(in_channels = out_channels * 6, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0,
                          bias=False)
    
    self.bnf = nn.BatchNorm2d(out_channels)
    
    self.adapool = nn.AdaptiveAvgPool2d(1)  
   
  
  def forward(self,x):

    x0 = self.conv0(x)
    x0 = self.bn0(x0)
    x0 = self.relu(x0)

    x1 = self.conv1(x)
    x1 = self.bn1(x1)
    x1 = self.relu(x1)
    
    x2 = self.conv2(x)
    x2 = self.bn2(x2)
    x2 = self.relu(x2)
    
    x3 = self.conv3(x)
    x3 = self.bn3(x3)
    x3 = self.relu(x3)
    
    x4 = self.conv4(x)
    x4 = self.bn4(x4)
    x4 = self.relu(x4)
    
    x5 = self.adapool(x)
    x5 = self.conv5(x5)
    x5 = self.bn5(x5)
    x5 = self.relu(x5)
    x5 = F.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear')

    x = torch.cat((x0,x1,x2,x3,x4,x5), dim = 1) #channels first
    x = self.convf(x)
    x = self.bnf(x)
    x = self.relu(x)
    
    return x
  
class DeepLabv3(nn.Module):
  
  def __init__(self, nc):
    
    super(DeepLabv3, self).__init__()
    
    self.nc = nc
    
    self.resnet_101 = torchvision.models.resnet101(pretrained = True)

    self.relu = nn.ReLU(inplace = True)
    
    self.assp = ASSP(in_channels = 1024)

    self.conv1 = nn.Conv2d(in_channels = 1280, out_channels = 256,
                          kernel_size = 1, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels = 768, out_channels = 256,
                          kernel_size = 1, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels = 512, out_channels = 256,
                          kernel_size = 1, stride=1, padding=0)    
      
    self.conv = nn.Conv2d(in_channels = 256, out_channels = self.nc,
                          kernel_size = 1, stride=1, padding=0)
    self.relu = nn.ReLU(inplace=True)
    self.bn = nn.BatchNorm2d(256)
  def forward(self,x):
    _, _, h, w = x.shape
    x = self.relu(self.resnet_101.bn1(self.resnet_101.conv1(x)))
    x = self.resnet_101.maxpool(x)
    x = self.resnet_101.layer1(x)
    layer1 = x
    x = self.resnet_101.layer2(x)
    layer2 = x
    x = self.resnet_101.layer3(x)
    layer3 = x
    x = self.assp(x)

    x = torch.cat((x,layer3),dim=1)
    x = self.conv1(x)
    x = self.bn(x)
    x = self.relu(x)

    x = F.interpolate(x, size=layer2.size()[2:], mode='bilinear')
    x = torch.cat((x,layer2),dim=1)
    x = self.conv2(x)
    x = self.bn(x)
    x = self.relu(x)

    x = F.interpolate(x, size=layer1.size()[2:], mode='bilinear')
    x = torch.cat((x,layer1),dim=1)
    x = self.conv3(x)
    x = self.bn(x)
    x = self.relu(x)

    x = self.conv(x)
    x = F.interpolate(x, size=(h, w), mode='bilinear') #scale_factor = 16, mode='bilinear')
    return x
 
sem_model = DeepLabv3(183).cuda()
