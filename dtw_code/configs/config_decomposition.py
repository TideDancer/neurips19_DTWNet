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
lr_dtw = 2e-3
lr_linear = 5e-3
alpha = 0.0
dtw_model = 'DTW_BOOSTING'


## synthetic datatset
dir = './'
dataset_name = 'synthetic'
downsample_rate = 0


## output file and redirection
output_file = 'log/boosting_'+dataset_name+'_model_'+dtw_model+'_ds_'+str(downsample_rate)+'_filter_'+str(n_filter)+'_lr_'+str(lr_dtw)
# sys.stdout = open(output_file, 'w')


## load dataset
train_loader, test_loader, sampleX, sampleY, feature_len = Util.load_boosting_mse(dir, dataset_name, batch_size, downsample_rate)


## model
kernel = torch.randn((5, n_filter, kernel_len))

dtw_list0 = []
dtw_list1 = []
dtw_list2 = []
dtw_list3 = []
dtw_list4 = []
for i in range(n_filter): 
  dtw_list0.append(eval('Model.'+dtw_model+'(kernel_len, feature_len, 1./n_filter, _kernel=kernel[0,i])'))
  dtw_list1.append(eval('Model.'+dtw_model+'(kernel_len, feature_len, 1./n_filter, _kernel=kernel[1,i])'))
  dtw_list2.append(eval('Model.'+dtw_model+'(kernel_len, feature_len, 1./n_filter, _kernel=kernel[2,i])'))
  dtw_list3.append(eval('Model.'+dtw_model+'(kernel_len, feature_len, 1./n_filter, _kernel=kernel[3,i])'))
  dtw_list4.append(eval('Model.'+dtw_model+'(kernel_len, feature_len, 1./n_filter, _kernel=kernel[4,i])'))
dtwlayer_list = [Model.DTWLAYER_BOOSTING(dtw_list0), Model.DTWLAYER_BOOSTING(dtw_list1), Model.DTWLAYER_BOOSTING(dtw_list2), Model.DTWLAYER_BOOSTING(dtw_list3), Model.DTWLAYER_BOOSTING(dtw_list4)]
model = Model.DTWNET_BOOSTING(n_class, dtwlayer_list)


## optimizer
optimizer = optim.Adam([{'params': model.dtwlayers.parameters(), 'lr': lr_dtw}], lr=5e-3)


## scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,500, 800], gamma=0.5)


## loss function
loss_func = Util.loss_function_mse


## training and testing
train_func = Util.train
test_func = Util.test

