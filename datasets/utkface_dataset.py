from torch.utils.data import Dataset
import torchvision.transforms as tfs
import os
from utils.align_utils import *
from PIL import Image
from utils.image_processing import ToTensor
import pandas as pd


class UTKDataset(Dataset):
    def __init__(self, data_root, csv_path, transform, resolution=256):
        self.data = pd.read_csv(csv_path)
        self.data_root = data_root
                               
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_makeup = self.data['makeup_name'][index]
        img_nomakeup = self.data['nomakeup_name'][index]
        makeup_age = self.data['makeup_age'][index]
        nomakeup_age = self.data['nomakeup_age'][index]
        gt_age = self.data['gt_age'][index]

        img_makeup = Image.open(os.path.join(self.data_root,img_makeup)).resize(
            (self.resolution, self.resolution))
        img_makeup = self.transform(img_makeup)

        img_nomakeup = Image.open(os.path.join(self.data_root,img_nomakeup)).resize(
            (self.resolution, self.resolution))
        img_nomakeup = self.transform(img_nomakeup)

        return img_makeup, img_nomakeup, gt_age, nomakeup_age


################################################################################

def get_utk_dataset(data_root, config):
    transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

    train_dataset = UTKDataset(data_root, './Datasets/UTKFace/annotations/utk_ann_train_cleaned.csv', transform, config.data.image_size)
    test_dataset = UTKDataset(data_root, './Datasets/UTKFace/annotations/utk_ann_test_cleaned.csv', transform, config.data.image_size)

    return train_dataset, test_dataset



