import sys
import numpy as np
import torch
from torch import nn, optim
import visdom
import Model
import time
import Util
import torch.utils.data
from torch.autograd import Variable
import sklearn
from sklearn.metrics import roc_auc_score
import time

def loss_function_mse(labels, predictions, **kwargs):
  loss = nn.MSELoss()
  return loss(predictions, labels)

def loss_function_mse_flat(labels, signalreduce, **kwargs):
  zero_sig = torch.zeros(signalreduce.shape)
  return loss_function_mse(zero_sig, signalreduce)

def loss_function_crossentropy(labels, predictions, **kwargs):
  loss = nn.CrossEntropyLoss()
  return loss(predictions, labels)

def train(epoch, batch_size, train_loader, model, optimizer, loss_func, **kwargs):
    model.train()
    train_loss = 0
    ft = 0
    bt = 0
    if 'scheduler' in kwargs: kwargs.get('scheduler').step()
    param_dict = kwargs.get('param_dict') if 'param_dict' in kwargs else None        
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data).float()
        label = Variable(torch.zeros((data.shape[0],1)))
        optimizer.zero_grad()
        ts = time.time()
        prediction = model(data)
        ft += time.time() - ts
        loss = loss_func(label, prediction, model=model, param_dict=param_dict)
        ts = time.time()
        loss.backward(retain_graph=True)
        bt += time.time() - ts
        #print('forward: ', time.time()-ts)
        optimizer.step()
        train_loss += loss.data.item()
        if batch_idx == len(train_loader.dataset)/batch_size-1:
          avg_loss = train_loss*batch_size/len(train_loader.dataset)
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
          # print('Frechet barycenter loss = {:.4f}'.format(avg_loss/data.shape[-1]))
          if 'vis' in kwargs: 
            kwargs.get('vis').line(X=torch.FloatTensor([epoch]),Y=torch.FloatTensor([train_loss*batch_size/len(train_loader.dataset)]), win='loss_train', update='append' if epoch > 0 else None)
            if loss_func == loss_function_mse: kwargs.get('vis').line(Y=prediction[0], win='prediction')
    # print('forward and backward time: ', ft, bt)

def test(epoch, test_loader, model, loss_func, **kwargs):
  if epoch % 10 == 0:
    model.eval()
    test_loss = 0
    test_auc = 0
    test_acc = 0
    correct = 0
    with torch.no_grad():
      for data, label in test_loader:
        data = Variable(data).float()
        if loss_func == loss_function_mse: label = Variable(label).type(torch.FloatTensor)
        else: label = Variable(label).type(torch.LongTensor)
        prediction = model(data)
        loss = loss_func(label, prediction)
        test_loss += loss.data.item()
        if loss_func == loss_function_mse: 
          test_acc += loss.data.item()
        else: 
          # test_auc += auc(label, prediction)
          _, pred_for_acc = torch.max(prediction.data, 1)
          test_acc += torch.sum(pred_for_acc==label.data)
          # test_acc += accuracy( prediction, label)
    print('test loss: ', test_loss)
    print('test auc: ' , test_auc)
    print('test acc: ' , float(test_acc)/len(test_loader.dataset))
    
    if 'vis' in kwargs:
      vis = kwargs.get('vis')
      vis.line(X=torch.FloatTensor([epoch]),Y=torch.FloatTensor([test_loss]), opts=dict(title='test loss',), win='loss_test', update='append' if epoch > 0 else None)
      vis.line(X=torch.FloatTensor([epoch]),Y=torch.FloatTensor([test_auc]), opts=dict(title='test auc',), win='auc_test', update='append' if epoch > 0 else None)
      vis.line(X=torch.FloatTensor([epoch]),Y=torch.FloatTensor([float(test_acc)/len(test_loader.dataset)]), opts=dict(title='test accuracy',), win='acc_test', update='append' if epoch > 0 else None)

def load_train(dir, dataset_name, batch_size, downsample_rate, **kwargs):
  X, Y, n_sample, feature_len = Util.read_ucr_downsample(dir+dataset_name+'/'+dataset_name+'_TRAIN', downsample_rate)
  if 'cat' in kwargs: 
    cat = int(kwargs.get('cat'))
    idx = list(np.where(Y==cat)[0])
    X = X[idx,:]
    Y = Y[idx]
  if 'size' in kwargs:
    size = int(kwargs.get('size'))
    if size > X.shape[0]: print('choose a smaller size value')
    else:
      X = X[:size, :]
      Y = Y[:size]
  train_loader = Util.proc_data(X, Y, batch_size)
  return train_loader, X, Y, feature_len

def load_test(dir, dataset_name, downsample_rate, **kwargs):
  X, Y, n_test_sample, feature_len = Util.read_ucr_downsample(dir+dataset_name+'/'+dataset_name+'_TEST', downsample_rate) 
  if 'cat' in kwargs: 
    cat = int(kwargs.get('cat'))
    idx = list(np.where(Y==cat)[0])
    X = X[idx,:]
    Y = Y[idx]
  # ## pick a subset for testing
  # test_sample_idx = np.random.randint(0,3000,size=(100,))
  # test_sample_idx = range(100)
  # X = X[test_sample_idx,:,:]
  # Y = Y[test_sample_idx]
  test_loader = Util.proc_data(X, Y, n_test_sample) # use full test data
  return test_loader, X, Y

## draw sample data for illustration
vis_enable = int(sys.argv[1])
if vis_enable == '1': vis = visdom.Visdom(env='model_barycenter')
verbose = int(sys.argv[2])
dataset_name = sys.argv[3]

## hyper params
batch_size = 4
# n_class = 5
n_filter = 1
lr_dtw = 5e-1
lr_linear = 5e-5
alpha = 0.0

## real dataset
dir = '/path/to/UCR_TS_Archive_2015/'
downsample_rate = 0

import pickle
import random
import dtwco

output_file = 'log/'+dataset_name+'.log'
sys.stdout = open(output_file, 'w')

train_loader, sampleX, sampleY, feature_len = load_train(dir, dataset_name, batch_size, downsample_rate)
classes = np.unique(sampleY)

kernel_len = int(feature_len/1.1)

dist_k = 0
for k in classes:

  train_loader, sampleX, sampleY, feature_len = load_train(dir, dataset_name, batch_size, downsample_rate, cat=k)
  # vis.line(Y=np.transpose(sampleX[0:4,0,:]), opts=dict(legend=list(sampleY[0:4]),), win='data0')
  
  kernel = torch.randn(kernel_len)
  dtw_list = [Model.DTW_FULL_DTWCO(kernel_len, feature_len, kernel=kernel)]
  dtwlayer_list = [Model.DTWLAYER(dtw_list)]
  model = Model.DTWNET_MSE(dtwlayer_list)
  
  optimizer = optim.Adam(model.parameters(), lr_dtw)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)
  loss_func = loss_function_mse
  
  for epoch in range(500):
    train(epoch, batch_size, train_loader, model, optimizer, loss_func, scheduler=scheduler)
    # vis.line(Y=model.dtwlayers[0].filters[0].kernel.data, opts=dict(legend=['filter'],), win='filter')
    for name, param in model.named_parameters():
      if verbose == '1':
        if param.requires_grad:
          # if 'kernel' in name: 
            # print(name, torch.sum(torch.abs((param.data))))
            if param.grad is not None:
              print(name, torch.sum(torch.abs((param.grad))))
            else:
              print(name, 'NO GRAD COMPUTED')
            # print(name, param.grad)
  
  #np.savetxt('log_ZZ', dtw_list[0].kernel.detach().numpy())
  Z = dtw_list[0].kernel.detach().numpy()

  print('finish training on class '+str(k))
  # _, sampleX, __ = load_test(dir, dataset_name, downsample_rate, cat=k)
  dist_total = 0
  for i in range(sampleX.shape[0]):
    dist, cost, path = dtwco.dtw(Z, sampleX[i], metric='sqeuclidean', dist_only=False)
    dist_total += dist
  dist_total = dist_total/sampleX.shape[0]
  dist_k += dist_total

  print('evaluating on class '+str(k)+' = '+str(dist_total))

dist_k = dist_k/classes.shape[0]

print(dataset_name, dist_k)

