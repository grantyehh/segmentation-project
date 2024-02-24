import torch
import torchvision
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
  
# data preprocess

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

mask_preprocess = transforms.Compose([
    transforms.Resize((224,224))
])

class Test(torch.utils.data.Dataset):
    
    def __init__(self, root, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.root = root
        self.image_path = os.path.join(self.root,'./test')
        self.sem_mask_path = os.path.join(self.root,'./testAns')
        self.ins_mask_path = os.path.join(self.root,'./test_combine3')

        self.fileimage = glob(self.image_path+'/*.jpg')
        self.file_sem_mask = glob(self.sem_mask_path+'/*.png')
        self.file_ins_mask = glob(self.ins_mask_path+'/*.png')

    def __len__(self): 
        return len(self.fileimage)     
        
    def __getitem__(self, idx):
        image = Image.open(self.fileimage[idx])
        sem_mask = Image.open(self.file_sem_mask[idx])
        ins_mask = Image.open(self.file_ins_mask[idx])
        
        if self.transform:
            image = self.transform(image)
        
        
        if self.target_transform:
            sem_mask = self.target_transform(sem_mask)
            ins_mask = self.target_transform(ins_mask)
        
        sem_mask = np.array(sem_mask)
        sem_mask = np.where(sem_mask==255,0,sem_mask)
        sem_mask = torch.from_numpy(sem_mask)

        ins_mask = np.array(ins_mask)
        ins_mask = np.where(ins_mask==255,0,ins_mask)
        ins_mask = torch.from_numpy(ins_mask)

        return image, sem_mask, ins_mask
    
    
test = Test('./coco_test',transform=preprocess,target_transform=mask_preprocess)    
print(len(test))
