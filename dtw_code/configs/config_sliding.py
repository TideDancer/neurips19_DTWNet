import sys
import numpy as np
import torch
from torch import nn, optim
import visdom
import Util
import Model
import time
 
## hyper params
batch_size = 10
n_class = 5
n_filter = 4
kernel_len = 10
lr_dtw = 5e-4
lr_linear = 5e-5
alpha = 0.0
dtw_model = 'DTW_FULL_DTWCO'
# dtw_model = 'DTW_FULL'

## real dataset
dir = '/path/to/UCR_TS_Archive_2015/'
dataset_name = sys.argv[4]
kernel_len = int(sys.argv[5])
stride = int(sys.argv[6])
downsample_rate = 0

## output file and redirection
output_file = 'log/'+dataset_name+'_model_'+dtw_model+'_ds_'+str(downsample_rate)+'_filter_'+str(n_filter)+'_lr_'+str(lr_dtw)+'_varkerlen'
# sys.stdout = open(output_file, 'w')

## load dataset
train_loader, sampleX, sampleY, feature_len, n_class = Util.load_train(dir, dataset_name, batch_size, downsample_rate)
test_loader, _, __ = Util.load_test(dir, dataset_name, downsample_rate)

## model
dtw_list = []
for i in range(n_filter): 
  dtw_list.append(eval('Model.'+dtw_model+'(kernel_len, feature_len)'))
in_chan = 1
dtwlayer_list = [Model.DTWLAYER_SHORTKERNEL(dtw_list, in_chan, feature_len, kernel_len, stride)]
model = Model.DTWNET_SINGLE(n_class, dtwlayer_list)
print(model)

## optimizer
optimizer = optim.Adam([{'params': model.dtwlayers[0].parameters(), 'lr': lr_dtw}, {'params':model.mlps.parameters(), 'lr': lr_linear}], lr=5e-3)
# optimizer = optim.Adam(model.parameters(), lr=lr_dtw)

## scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600], gamma=0.5)

## loss function
loss_func = Util.loss_function_crossentropy

## training and testing
train_func = Util.train
test_func = Util.test

