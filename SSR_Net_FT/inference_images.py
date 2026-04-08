"""Description for the script:
inference one single image or many images of a directory using pretrained SSRNet.
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from SSR_models.SSR_Net_model import SSRNet
import argparse
import time

import numpy as np
import torch
from torchvision import transforms as T
import cv2
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inference_single_image(model_, image_path_, input_size_=64):
    image_ = cv2.imread(image_path_)
    start_time_ = time.time()
    image_ = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size_, input_size_)),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(image_)
    
    image_ = image_[np.newaxis, ]
    image_ = image_.to(device)
    with torch.set_grad_enabled(False):
        results_ = model_(image_)
    return results_,  time.time() - start_time_


if __name__ == "__main__":
    image_file_path = "../../datasets/megaage_asion/megaage_asian/megaage_asian/test/13.jpg"
    model_file = "../checkpoint/ssrnet_finetuned.pth"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed, dir or a single image.")
    parser.add_argument("--graph", help="graph/model to be executed")
    
    args = parser.parse_args()
    
    if args.graph:
        model_file = args.graph
    if args.image:
        image_file_path = args.image
    
    input_size = 64
    
    inference_model = SSRNet()
    loaded_model = torch.load(model_file)
    inference_model.load_state_dict(loaded_model['state_dict'])
    inference_model = inference_model.to(device)
    inference_model.eval()
    
    if os.path.isfile(image_file_path):  # inference a single image
        age_, cost_time = inference_single_image(inference_model, image_file_path)
        print("age:\t{}, used {} s in total.".format(age_[0], cost_time))
    elif os.path.isdir(image_file_path):  # a directory containing many images, inference them all!
        results_list = []
        for image in os.listdir(image_file_path):
            try:
                age_, _ = inference_single_image(inference_model, os.path.join(image_file_path, image))
                results_list.append([image,np.round(age_[0].cpu().numpy()).astype(int)])
                print("age:\t{}\t, image:\t{}".format(age_.tolist()[0], image))
            except Exception as e:
                print("Error: ", image)
                continue
        
        
        # just a glimpse of the predicted results.
        pd_result = pd.DataFrame(results_list, columns=['img_name','age'])
        pd_result.to_csv('./results/ssrnet_preds_utk_makeup_clean_2204.csv',  index=False)
        
