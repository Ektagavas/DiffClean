"""Description for the script:
construct a dataloader for UTK datasets.
"""

import cv2
import os

from torch.utils.data import Dataset
import numpy as np
np.bool = bool
from imgaug import augmenters as iaa
from torchvision import transforms as T
import pandas as pd
from tqdm import tqdm
import json
import PIL.Image
import PIL.ImageFile

import scipy.ndimage
from tqdm import tqdm


class UTKDataset(Dataset):
    def __init__(self, image_csv_path='utk_age_data.csv', base_path='', augment=False, mode='train', input_size=64):
        self.base_path = base_path
        self.image_csv_path = os.path.join(base_path, image_csv_path)
        self.mode = mode
        self.augment = augment
        self.input_size = input_size
        self.imagelist = []
        self.process_paths()

    def process_paths(self):
        print('Processing images')
        data = pd.read_csv(self.image_csv_path)
        missing_images = 0
        
        for i, row in tqdm(data.iterrows()):
            imgpath = os.path.join(self.base_path,row['img_name'])
            if os.path.exists(imgpath):
                try:
                    # im_cv = cv2.imread(imgpath)
                    # im_h, im_w = im_cv.shape[:2]
                    self.imagelist.append([imgpath, row['age'], row['age_group']])
                except Exception:
                    missing_images += 1
                    print('Missing: ', imgpath)
            else:
                print('Missing: ', imgpath)
                missing_images += 1
        print(f'Number of missing images: {missing_images}/{len(data)}')
        

    
    def __len__(self):
        return len(self.imagelist)
    
    
    def __getitem__(self, index):
        image_, image_path_ = self.read_images(index)      
        if self.mode in ['train', ]:
            label = int(self.imagelist[index][1])
        else:
            label = image_path_
        if self.augment:
            image_ = self.augmentor(image_)
        image_ = T.Compose([
            T.ToPILImage(),
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(image_)
        return image_.float(), label, image_path_, self.imagelist[index][2]
    
    def read_images(self, index_):
        image_path_ = self.imagelist[index_][0]
        image = cv2.imread(image_path_)
        return image, image_path_
    
    def augmentor(self, image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
        ], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug


if __name__ == "__main__":
    dataset = UTKDataset(image_csv_path='utk_age_train_bal.csv',base_path='./Datasets/UTKFace')
    j = 0
    age_groups = [(0,2), (3,6), (7,9), (10,14), (15,19), (20,29), (30,39), (40,49), (50,69), (70,float('inf'))]
    names = ["(0;2)", "(3;6)", "(7;9)", "(10;14)", "(15;19)", "(20;29)", "(30;39)", "(40;49)", "(50;69)", "(70;inf)"]
    from collections import defaultdict
    d = defaultdict(int)
    def get_age_group(age):
        for index, (start, end) in enumerate(age_groups):
            if start <= age <= end:
                return index
    for i, (image, label,_) in tqdm(enumerate(dataset)):
        j += 1
        d[names[get_age_group(label)]] += 1
    print(d)
