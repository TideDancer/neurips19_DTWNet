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
lr_dtw = 1e-2
lr_linear = 5e-4
# alpha = 0.0
# dtw_model = 'DTW_SPRING_EPS'
dtw_model = 'DTW_FULL_DTWCO'
# dtw_model = 'DTW_FULL'

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
dtw_list = []
for i in range(n_filter): 
  dtw_list.append(eval('Model.'+dtw_model+'(kernel_len, feature_len)'))
dtwlayer_list = [Model.DTWLAYER(dtw_list)]
model = Model.DTWNET_SINGLE(n_class, dtwlayer_list)


## optimizer
optimizer = optim.Adam([{'params': model.dtwlayers.parameters(), 'lr': lr_dtw}, {'params':model.mlps.parameters(), 'lr': lr_linear}], lr=5e-3)


## scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.1)

# ################ MLP model#################
# model = Model.MLP(3, feature_len, n_class)
# optimizer = optim.Adam(model.parameters(), lr=lr_linear)
# #############################################

## loss function
loss_func = Util.loss_function_crossentropy
# loss_func = Util.loss_function_crossentropy_shape

## training and testing
train_func = Util.train
test_func = Util.test

