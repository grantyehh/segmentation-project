import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from glob import glob
import os

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

mask_preprocess = transforms.Compose([
    transforms.Resize((224,224))
])

class COCO_Train(torch.utils.data.Dataset):
    
    def __init__(self, root, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.root = root
        self.mask_path = os.path.join(self.root,'./trainAns')
        self.image_path = os.path.join(self.root,'./train')

        self.fileimage = glob(self.image_path+'/*.jpg')
        self.filemask = glob(self.mask_path+'/*.png')

    def __len__(self): 
        return len(self.fileimage)     
        
    def __getitem__(self, idx):
        image = Image.open(self.fileimage[idx])
        mask = Image.open(self.filemask[idx])
        
        if self.transform:
            image = self.transform(image)
        
        
        if self.target_transform:
            mask = self.target_transform(mask)
        
        mask = np.array(mask)
        mask = np.where(mask==255,0,mask)
        mask = torch.from_numpy(mask)

        return image, mask
    
class COCO_Val(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.root = root
        self.mask_path = os.path.join(self.root,'./valAns')
        self.image_path = os.path.join(self.root,'./val')

        self.fileimage = glob(self.image_path+'/*.jpg')
        self.filemask = glob(self.mask_path+'/*.png')

    def __len__(self):
        return len(self.fileimage)     
        
    def __getitem__(self, idx):
        image = Image.open(self.fileimage[idx])
        mask = Image.open(self.filemask[idx])
        
        if self.transform:
            image = self.transform(image)
        
        
        if self.target_transform:
            mask = self.target_transform(mask)
        
        mask = np.array(mask)
        mask = np.where(mask==255,0,mask)
        mask = torch.from_numpy(mask)

        return image, mask

# Data    
train = COCO_Train('./coco_test',transform=preprocess,target_transform=mask_preprocess)
val = COCO_Val('./coco_test',transform=preprocess,target_transform=mask_preprocess)
# print(len(train))
# print(len(val))

# Loader
from torch.utils.data import DataLoader
trainloader = DataLoader(train,batch_size=8, num_workers=2, shuffle=False)
valloader = DataLoader(val,batch_size=8, num_workers=2, shuffle=False)

# Model
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASSP(nn.Module):
  def __init__(self,in_channels,out_channels = 256):
    super(ASSP,self).__init__()
    
    
    self.relu = nn.ReLU(inplace=True)
    
    self.conv1 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0,
                          bias=False)
    
    self.bn1 = nn.BatchNorm2d(out_channels)
    
    self.conv2 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 2,
                          dilation = 2,
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
                          padding = 6,
                          dilation = 6,
                          bias=False)
    
    self.bn4 = nn.BatchNorm2d(out_channels)
    
    self.conv5 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0,
                          bias=False)
    
    self.bn5 = nn.BatchNorm2d(out_channels)



    self.convf = nn.Conv2d(in_channels = out_channels * 5, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride = 1,
                          padding = 0,
                          bias=False)
    
    self.bnf = nn.BatchNorm2d(out_channels)
    
    self.adapool = nn.AdaptiveAvgPool2d(1)  
   
  
  def forward(self,x):
    
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

    x = torch.cat((x1,x2,x3,x4,x5), dim = 1) #channels first
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
  
import segmentation_models_pytorch as smp
device = torch.device("cuda")
from torch.optim import Adam

model = DeepLabv3(183).to(device)
#criterion = nn.CrossEntropyLoss()
criterion = smp.losses.DiceLoss(mode='multiclass')
optimizer = Adam(model.parameters(),lr=1e-4)
epochs = 50

# class DeepLabCE(nn.Module):

#     def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
#         super(DeepLabCE, self).__init__()
#         self.top_k_percent_pixels = top_k_percent_pixels
#         self.ignore_label = ignore_label
#         self.criterion = nn.CrossEntropyLoss(weight=weight,
#                                              ignore_index=ignore_label,
#                                              reduction='none')

#     def forward(self, logits, labels):
#         pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
#         if self.top_k_percent_pixels == 1.0:
#             return pixel_losses.mean()

#         top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
#         pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
#         return pixel_losses.mean()
    
from tqdm import tqdm
trainLossHis,valLossHis = [], []
#model.load_state_dict(torch.load('./deeplab-newtrain.pt'))
if __name__ == '__main__':
    for e in range(epochs):
        print("Epoch {}".format(e+1))
        running_loss = 0.0
        val_loss = 0.0
        # training set
        model.train()
        for data in tqdm(trainloader):
            img ,label = data
            img = img.to(device)
            label = label.long().to(device)
            output= model(img)

            loss = criterion(output,label)    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
        # validation set
        model.eval()
        with torch.no_grad():
            for data in tqdm(valloader):
                img ,label = data
                img = img.to(device)
                label = label.long().to(device)        
                output= model(img)  

                loss = criterion(output,label)  
                val_loss+=loss.item()

        trainLossHis.append(running_loss/len(trainloader))
        valLossHis.append(val_loss/len(valloader))

        # plt.plot(range(len(trainLossHis)), trainLossHis, label='train loss')
        # plt.plot(range(len(valLossHis)), valLossHis, label='validation loss')
        # plt.legend()
        # plt.show()
        print('Train loss: {}'.format(trainLossHis[-1]))
        print('Val loss: {}'.format(valLossHis[-1]))
        torch.save(model.state_dict(),'./deeplab-246.pt')




