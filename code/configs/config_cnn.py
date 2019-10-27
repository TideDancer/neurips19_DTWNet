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
n_class = 2
n_filter = 1
kernel_len = 10
lr_dtw = 2e-4
lr_linear = 2e-4
alpha = 0.0
dtw_model = 'CONV_MLP'


## synthetic datatset
dir = './'
dataset_name = 'synthetic'
downsample_rate = 0


## output file and redirection
output_file = 'log/'+dataset_name+'_model_'+dtw_model+'_ds_'+str(downsample_rate)+'_filter_'+str(n_filter)+'_lr_'+str(lr_dtw)
# sys.stdout = open(output_file, 'w')


## load dataset
train_loader, sampleX, sampleY, feature_len, _ = Util.load_train(dir, dataset_name, batch_size, downsample_rate)
test_loader, _, __ = Util.load_test(dir, dataset_name, downsample_rate)


## model
model = Model.CONV_MLP(n_class, feature_len, n_filter, kernel_len)


## optimizer
optimizer = optim.Adam(model.parameters(), lr_linear)


## scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.3)


## loss function
loss_func = Util.loss_function_crossentropy


## training and testing
train_func = Util.train
test_func = Util.test

