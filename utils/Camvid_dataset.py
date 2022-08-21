# 导入库
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
 
torch.manual_seed(17)

class CamVidDataset(torch.utils.data.Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    def __init__(self, images_dir,root):
        self.transform = A.Compose([
            A.Resize(448, 448),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])
        
        list_total=[]
        list_img = []
        list_mask = []
        with open(images_dir,'r') as f :
            for line in f.readlines():
                line=line.strip('\n')
                list_total.append(line)
        for pair in list_total:
            p = pair.split(' ')
            img = p[0]
            mask = p[1]
            list_img.append(img)
            list_mask.append(mask)
        self.root = root
        self.ids = len(list_img)
        # self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.images_fps = list_img
        self.masks_fps = list_mask
    
    def __getitem__(self, i):
        # read data
        img_path = os.path.join(self.root,self.images_fps[i][1:])
        mask_path = os.path.join(self.root,self.masks_fps[i][1:])
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array( Image.open(mask_path).convert('RGB'))
        image = self.transform(image=image,mask=mask)
        
        return image['image'], image['mask'][:,:,0]
        
    def __len__(self):
        return self.ids
    
    
# # 设置数据集路径
# DATA_DIR = r'database/camvid/camvid/' # 根据自己的路径来设置

    
# train_dataset = CamVidDataset("E:/DATABASE/DAVIS/ImageSets/480p/train.txt",'E:/DATABASE/DAVIS')
# val_dataset = CamVidDataset("E:/DATABASE/DAVIS/ImageSets/480p/trainval.txt",'E:/DATABASE/DAVIS')

 
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True,drop_last=True)
# for idx,data in enumerate(train_loader):
#     print(idx,data)
#     break