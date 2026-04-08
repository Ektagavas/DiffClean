"""Description for the script:
train SSR-Net.
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets.read_utk_data import UTKDataset
from SSR_models.SSR_Net_model import SSRNet
from utils.loss import AdjustSmoothL1Loss
from utils.misc_utils import EarlyStopper
from datetime import datetime
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model_, dataloaders_, criterion_, optimizer_, num_epochs_=25,ckptdir='checkpoint'):
    global lr_scheduler
    
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = 0.0

    loss_early_stopper = EarlyStopper(patience=10, min_delta=0.001)
    is_stopping = False
    epoch = 0


    while epoch < num_epochs_ and not is_stopping:
        print('\nEpoch {}/{}'.format(epoch, num_epochs_ - 1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model_.train()  # Set model to training mode
                print('in train mode...')
            else:
                print('in {} mode...'.format(phase))
                model_.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects_3 = 0
            running_corrects_5 = 0
            running_agg = 0
            running_total = 0
            for i, (inputs, labels, _, agegroups) in enumerate(dataloaders_[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                
                # zero the parameter gradients
                optimizer_.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_(inputs)
                    if criterion_.__class__.__name__ == 'AdjustSmoothL1Loss':
                        loss = criterion_(outputs, labels, agegroups)
                    else:
                        loss = criterion_(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer_.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects_3 += torch.sum(torch.abs(outputs - labels) < 3)  # CA 3
                running_corrects_5 += torch.sum(torch.abs(outputs - labels) < 5)  # CA 5

            
            epoch_loss = running_loss / len(dataloaders_[phase].dataset)
            CA_3 = running_corrects_3.double() / len(dataloaders_[phase].dataset)
            CA_5 = running_corrects_5.double() / len(dataloaders_[phase].dataset)
            
            
            print('{} Loss: {:.4f} CA_3: {:.4f}, CA_5: {:.4f}'.format(phase, epoch_loss, CA_3, CA_5))
            time_elapsed = time.time() - since
            print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60),flush=True)
            
            # deep copy the model
            if phase == 'val' and CA_3 > best_acc:
                best_acc = CA_3
                best_model_wts = copy.deepcopy(model_.state_dict())
            if phase == 'val':
                val_acc_history.append(CA_3)
                if loss_early_stopper(epoch_loss):
                    print(f"Early stopping triggered at epoch: {epoch} {loss_early_stopper.early_stop}")
                    is_stopping = True
                    break
        
        lr_scheduler.step()

        if epoch % 10 == 0:
            torch.save({'epoch': epoch,'state_dict': best_model_wts,#model_to_train.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),}, 
            os.path.join(ckptdir,'model_Adam_{}_delta_{}_LRDecay_weightDecay{}_batch{}_lr{}_epoch{}_64x64.pth'.format(criterion.__class__.__name__,delta,weight_decay, batch_size, learning_rate, epoch)))
            print('Best model saved: ')
        epoch += 1
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val CA_3: {:4f}'.format(best_acc))
    
    # load best model weights
    model_.load_state_dict(best_model_wts)
    return model_, val_acc_history


if __name__ == "__main__":
    batch_size = 50
    input_size = 64
    num_epochs = 200 
    learning_rate = 1e-3.  #0.0015  # originally 0.001
    weight_decay = 1e-4
    load_pretrained = True

    model_file = '../checkpoint/ssrnet_finetuned.pth'

    model_to_train = SSRNet(image_size=input_size)
    if load_pretrained:
        print('Model loaded from ', model_file)
        loaded_model = torch.load(model_file, weights_only=False)
        model_to_train.load_state_dict(loaded_model['state_dict'])
    

    train_gen = UTKDataset(image_csv_path='annotations/utk_ann_train_full.csv',base_path='./Datasets/UTKFace', augment=True, mode="train")
    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6)
    
    val_gen = UTKDataset(image_csv_path='annotations/utk_ann_test_full.csv',base_path='./Datasets/UTKFace', augment=False, mode="train")
    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=6)

    
    
    total_dataloader = {
        'train': train_loader,
        'val': val_loader,
        # 'test': test_loader,
    }
    
    model_to_train = model_to_train.to(device)
    
    params_to_update = model_to_train.parameters()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    delta=1.0
    
    criterion = AdjustSmoothL1Loss()
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=num_epochs)
    
    ckptdir = os.path.join('checkpoint', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(ckptdir,exist_ok=True)

    # Train and evaluate
    model_to_train, hist = train_model(model_to_train, total_dataloader, criterion, optimizer_ft,
                                       num_epochs_=num_epochs, ckptdir=ckptdir)
    
    # Saving best model
    torch.save({
        'epoch': num_epochs,
        'state_dict': model_to_train.state_dict(),
        'optimizer_state_dict': optimizer_ft.state_dict(),
    }, os.path.join(ckptdir,'best_model_Adam_{}_delta_{}_LRDecay_weightDecay{}_batch{}_lr{}_nepochs{}_64x64.pth'.format(criterion.__class__.__name__,
            delta,weight_decay, batch_size, learning_rate, num_epochs)))
    