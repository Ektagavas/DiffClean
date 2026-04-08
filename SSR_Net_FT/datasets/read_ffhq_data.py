"""Description for the script:
construct a dataloader for FFHQ datasets.
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


class FFHQDataset(Dataset):
    def __init__(self, image_csv_path='ffhq_age_data.csv', base_path='', augment=False, mode='train', input_size=64):
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
            imgpath = os.path.join(self.base_path, row['img_name'])
            if os.path.exists(imgpath):
                try:
                    # im_cv = cv2.imread(imgpath)
                    # im_h, im_w = im_cv.shape[:2]
                    self.imagelist.append([imgpath, row['age'],row['age_group']])
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
        # image = PIL.Image.open(image_path_)
        # aligned = self.align_in_the_wild_image(image,filename)
        # return aligned, image_path_
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
    
    def align_in_the_wild_image(self,img, filename, output_size=256, transform_size=4096, enable_padding=True):
        idx = str(int(filename.rsplit('/',1)[1].split('.')[0]))
        spec = self.json_data[idx]['image']
        # item_idx = int(os.path.basename(spec['file_path'])[:-4])

        # Parse landmarks.
        # pylint: disable=unused-variable
        lm = np.array(spec['face_landmarks'])
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 2.2) # This results in larger crops then the original FFHQ. For the original crops, replace 2.2 with 1.8
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        # src_file = spec['file_path']
        # if not os.path.isfile(src_file):
            # print('\nCannot find source image. Please run "--wilds" before "--align".')
            # return
        # img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.Resampling.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

        # Save aligned image.
        # dst_subdir = os.path.join(dst_dir, '%05d' % (item_idx - item_idx % 1000))
        # os.makedirs(dst_subdir, exist_ok=True)
        # img.save(os.path.join(dst_subdir, '%05d.png' % item_idx))
        return img


if __name__ == "__main__":
    dataset = FFHQDataset(image_csv_path='ffhq_age_train_bal.csv',base_path='./Datasets/FFHQ_Aging_Filtered')
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
