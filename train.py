import os
import pdb

import tsaug

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import pandas as pd
import numpy as np
import csv
import torch
import random
import torch.utils.data as data
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from tensorboardX import SummaryWriter

writer = SummaryWriter()

import time

import process
from dataloader import GaitDataset
from model import BMRRConvNet
from tslearn.preprocessing import TimeSeriesResampler

import argparse
parser = argparse.ArgumentParser()
from initial import initial
args = initial(parser)

# set ramdom_seed
seed = args.seed_select

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.set_deterministic(True)

# set hyperparameters
epochs = args.epoch
batch_size = args.batch_size
learning_rate = args.lr
step_gamma = args.step_gamma
train_log_interval = args.train_interval
target_classes = args.target_classes
select_code = args.code
# set model root and name for saving
MODEL_PATH = args.dir_checkpoint
NAME = args.dir_checkpoint_name

start_time = time.time()

# data loader and to tensor
# train-set part
print("processing for train-set--")
files_train, labels_train, mean_train, std_train, names_train = process.init(root=args.dir_train)
aug_data_train, aug_label_train = process.getData(files=files_train, labels=labels_train,
                                                  mean=mean_train, std=std_train, times_aug=30)


dataset_train = GaitDataset(datas=aug_data_train, labels=aug_label_train, names=names_train)
dataloader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=False)
num_batches_t = len(dataloader_train)

# validation-set part
print("processing for validation-set--")
files_val, labels_val, mean_val, std_val, names_val = process.init(root=args.dir_validation)
aug_data_val, aug_label_val = process.getData(files=files_val, labels=labels_val,
                                              mean=mean_train, std=std_train, times_aug=0)

dataset_val = GaitDataset(datas=aug_data_val, labels=aug_label_val, names=names_val)
dataloader_val = DataLoader(dataset=dataset_val,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False)
num_batches_v = len(dataloader_val)

print("Number of batches of training dataset : ", num_batches_t)
print("Number of batches of validation dataset : ", num_batches_v)

#if not (len(dataset_train) == len(dataset_val)):
#    print("Error: Numbers of class in training set and validation set are not equal")
#    exit()

# set gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set model
custom_model = BMRRConvNet(num_classes=target_classes).to(device)
#summary(custom_model)

# set loss function, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=learning_rate, weight_decay=0.001)
#optimizer = torch.optim.SGD(custom_model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = StepLR(optimizer, step_size=1, gamma=step_gamma)

# define train and validation function
def train(model, dataloader_train, dataloader_validation, optimizer):
    model.train()

    # for train
    train_loss_total = 0
    train_corr_num = 0
    train_total_num = 0

    n = 1
    print(">> training")
    for batch_idx, (data_tr, label_tr) in enumerate(dataloader_train):
        data_tr, label_tr = data_tr.to(device), label_tr.to(device)

        data_tr = data_tr.float()
        data_tr = data_tr.transpose(1, 2)

        train_outputs = model(data_tr)
        optimizer.zero_grad()
        model_predicted_tr = train_outputs.argmax(dim=1)

        train_corr = label_tr[label_tr == model_predicted_tr].size(0)
        train_corr_num += train_corr
        train_total_num += label_tr.size(0)

        train_loss = criterion(train_outputs, label_tr)
        train_loss.backward()
        optimizer.step()
        train_loss_total += train_loss.item()

        del train_outputs
        del train_loss

    train_loss_result = train_loss_total / train_total_num
    train_accuracy_result = (train_corr_num / train_total_num) * 100

    ## validation part
    # for validation
    val_loss_total = 0
    val_corr_num = 0
    val_total_num = 0
    val_loss_result = 0
    val_accuracy_result = 0

    model.eval()
    print(">> validating")
    with torch.no_grad():
        for batch_idx, (data_v, label_v) in enumerate(dataloader_validation):
            data_v, label_v = data_v.to(device), label_v.to(device)

            data_v = data_v.float()
            data_v = data_v.transpose(1, 2)

            val_outputs = model(data_v)
            model_predicted_v = val_outputs.argmax(dim=1)

            val_corr = label_v[label_v == model_predicted_v].size(0)
            val_corr_num += val_corr
            val_total_num += label_v.size(0)

            val_loss = criterion(val_outputs, label_v)
            val_loss_total += val_loss.item()

            del val_outputs
            del val_loss

        val_loss_result = val_loss_total / val_total_num
        val_accuracy_result = (val_corr_num / val_total_num) * 100

    return train_loss_result, train_accuracy_result, val_loss_result, val_accuracy_result

# define for saving checkpoints
def save_checkpoint(epoch, model, batch, optimizer, loss, filename):
    state = {
        'Epoch': epoch,
        'State_dict': model.state_dict(),
        'Optimizer': optimizer.state_dict(),
        'Loss': loss,
        'Batch': batch
    }
    torch.save(state, filename)

# set variables for train, validation
train_loss_list = []
validation_loss_list = []

loss_min = np.inf
loss_min_epoch = 0
accuracy_max = 0
accuracy_max_epoch = 0

early_stop = 0

# part of training & validating
##Training & Validating
for epoch in range(epochs):
    print("\n===================================")
    print("Current epochs : " + str(epoch + 1))
    print("=====================================")
    current_epoch = epoch + 1

    train_loss, train_accuracy, val_loss, val_accuracy = train(model=custom_model,
                                                               dataloader_train=dataloader_train,
                                                               dataloader_validation=dataloader_val,
                                                               optimizer=optimizer)

    print(">> Epoch: {}/{} | Training loss: {:.4f} | Validation loss: {:.4f} | Training Acc: {:.4f}% | Validation Acc: {:.4f}%".
          format(epoch + 1,
                 epochs,
                 train_loss,
                 val_loss,
                 train_accuracy,
                 val_accuracy))

    #train_loss_list.append(train_loss_total / (train_log_interval * 7))
    #validation_loss_list.append(val_loss_total / num_batches_val)

    if select_code == 0:
        if val_loss < loss_min:
            loss_min = val_loss
            loss_min_epoch = current_epoch

        if val_accuracy > accuracy_max:
            print("\n***************************************************************************************************************** best model!")
            accuracy_max = val_accuracy
            accuracy_max_epoch = current_epoch

            early_stop = 0
        else:
            early_stop += 1
    elif select_code == 1:
        if val_accuracy > accuracy_max:
            accuracy_max = val_accuracy
            accuracy_max_epoch = current_epoch

        if val_loss < loss_min:
            print("\n***************************************************************************************************************** best model!")
            loss_min = val_loss
            loss_min_epoch = current_epoch

            early_stop = 0
        else:
            early_stop += 1
    '''
        A = str(epoch + 1) + NAME
        print("Saved model = ", MODEL_PATH + A)
        save_checkpoint(epoch=current_epoch,
                        model=custom_model,
                        batch=batch_size,
                        optimizer=optimizer,
                        loss=val_loss,
                        filename=MODEL_PATH + A)
        early_stop = 0
    else:
        early_stop += 1
    '''

    A = str(epoch + 1) + NAME
    print("Saved model = ", MODEL_PATH + A)
    save_checkpoint(epoch=current_epoch,
                    model=custom_model,
                    batch=batch_size,
                    optimizer=optimizer,
                    loss=val_loss,
                    filename=MODEL_PATH + A)

    print("\n>>>")
    print("early_stop: ", early_stop)
    if early_stop > 9:
        print("\nEarly stop!")
        print("Total Elapsed Time : {} s".format(time.time() - start_time))

        print("\nEpoch for best accuracy during training: {} | {}".format(accuracy_max_epoch, accuracy_max))
        print("Epoch for best loss during training: {} | {}".format(loss_min_epoch, loss_min))

        exit()
    '''
    if current_epoch == 100:
        scheduler.step()
        print("//////////////////////////////////////////////////////////////////////////////////////step up//////////")
   
    if current_epoch % 60 == 0:
        scheduler.step()
        print("//////////////////////////////////////////////////////////////////////////////////////step up//////////")
    '''
print('\nTraining Complete')
print("Total Elapsed Time : {} s".format(time.time() - start_time))

print("=========================================================================")
