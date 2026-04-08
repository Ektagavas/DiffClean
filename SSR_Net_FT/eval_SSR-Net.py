"""Description for the script:
train SSR-Net.
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.read_ffhq_data import FFHQDataset
from datasets.read_utk_data import UTKDataset
from SSR_models.SSR_Net_model import SSRNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_model(model_, testloader):
    model_.eval()
    since = time.time()
    
    running_corrects_3 = 0
    running_corrects_5 = 0
    running_corrects = 0
    running_mae = 0
    pathlist = []
    preds = []
    gt = []
    for i, (inputs, labels, paths, _) in enumerate(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model_(inputs)

        preds.extend(list(torch.round(outputs).detach().cpu().numpy()))
        pathlist += paths
        
        # statistics
        # print(torch.round(outputs), labels)
        running_corrects += torch.sum(torch.round(outputs)==labels)
        running_corrects_3 += torch.sum(torch.abs(outputs - labels) < 3)  # CA 3
        running_corrects_5 += torch.sum(torch.abs(outputs - labels) < 5)  # CA 5
        running_mae += torch.sum(torch.abs(outputs - labels))
    
    acc = running_corrects.double() / len(testloader.dataset)
    mae = running_mae.double() / len(testloader.dataset)
    CA_3 = running_corrects_3.double() / len(testloader.dataset)
    CA_5 = running_corrects_5.double() / len(testloader.dataset)

    res = np.hstack((np.array(pathlist).reshape(-1,1), np.array(preds).reshape(-1,1)))
    np.savetxt('./results/pretrained_utk_'+os.path.basename(model_file)+'.csv', res,fmt='%s', delimiter=',',header='img_name,age',comments='')
    
    
    print('MAE: {:.4f} Acc: {:.4f} CA_3: {:.4f}, CA_5: {:.4f}'.format(mae, acc, CA_3, CA_5))
    
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    
    batch_size = 50
    input_size = 64
    augment = False

    model_file = '../checkpoint/ssrnet_finetuned.pth'
    
    
    model = SSRNet(image_size=input_size)
    print('Model loaded from ', model_file)
    loaded_model = torch.load(model_file, weights_only=False)
    model.load_state_dict(loaded_model['state_dict'])


    test_gen = UTKDataset(image_csv_path='annotations/utk_ann_test_full.csv',base_path='./Datasets/UTKFace', augment=False, mode="train")
    test_loader = DataLoader(test_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    
    model = model.to(device)
    
    # Evaluate
    eval_model(model, test_loader)
