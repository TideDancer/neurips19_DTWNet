import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import sklearn
from sklearn.metrics import roc_auc_score
import time


## auc and accuracy
def auc(y_true, y_score):
  return roc_auc_score(y_true, y_score[:,1])

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0]

## loss functions
def loss_function(labels, predictions):
  loss = nn.MSELoss()
  return loss(predictions, labels)

def loss_function_multi(labels, predictions, **kwargs):
  num_filter = kwargs.get('params')['num_filter']
  predictions = predictions.view(predictions.shape[1], -1)
  loss = nn.MSELoss()
  total = 0
  for i in range(num_filter):
    total += loss(predictions[i], labels[i])
  return total

def loss_function_crossentropy(labels, predictions, **kwargs):
  loss = nn.CrossEntropyLoss()
  return loss(predictions, labels)

def loss_function_crossentropy_diverse(labels, predictions):
  l = nn.CrossEntropyLoss()
  loss_ce = l(predictions,labels)
  loss_di = torch.abs(torch.dot(model.dtw[0].filters[0].kernel, model.dtw[0].filters[1].kernel))
  #loss_di = torch.abs(torch.dot(model.conv.weight[0,0], model.conv.weight[1,0]))
  alpha = 1e-1
  return (1-alpha)*loss_ce + alpha*loss_di

def loss_function_crossentropy_shape(labels, predictions, model, param_dict=None, alpha=0.1): # need to fix later
  l = nn.CrossEntropyLoss()
  loss_ce = l(predictions,labels)
  # loss_shape = ((model.dtw[0].filters[0].kernel)[0] - (model.dtw[0].filters[0].kernel)[-1])**2
  ## need fix
  loss_shape2 = (model.dtwlayers[0].filters[0].kernel)[0]**2
  loss_shape3 = (model.dtwlayers[0].filters[0].kernel)[-1]**2
  return (1-alpha)*loss_ce + alpha*(loss_shape2+loss_shape3)

def loss_function_mse(labels, predictions, **kwargs):
  loss = nn.MSELoss()
  return loss(predictions, labels)

def loss_function_mse_flat(labels, signalreduce, **kwargs):
  zero_sig = torch.zeros(signalreduce.shape)
  return loss_function_mse(zero_sig, signalreduce)

def loss_function_hybrid(labels, signalreduce, predictions, alpha=10, **kwargs):
  mseloss = loss_function_mse_flat(labels, signalreduce)
  clsloss = loss_function_crossentropy(labels, predictions)
  return alpha*mseloss + clsloss


## dataset read util
def read_ucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    Y = list(map(lambda e: e-1, Y))
    Y = np.asarray(Y)
    X = data[:,1:]
    X = np.reshape(X, (X.shape[0], 1, -1)) # shape is (N, channel, L)
    return X, Y, X.shape[0], X.shape[2]

def read_ucr_downsample(filename, sample_rate):
    # if sample_rate == 0, no downsample
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    Y = np.asarray(Y)
    X = data[:,1:]
    if sample_rate > 0:
      X = X[:, ::sample_rate]
    X = np.reshape(X, (X.shape[0], 1, -1)) # shape is (N, channel, L)
    return X, Y, X.shape[0], X.shape[2]

class UCRDataset(torch.utils.data.Dataset): ## need to fix later
  def __init__(self, file_path):
    X, Y, self.count, self.input_len = read_ucr(file_path)
    self.data = torch.from_numpy(X).double()
    self.label= torch.from_numpy(Y)

  def __getitem__(self, index):
    return (self.data[index], self.label[index])

  def __len__(self):
    return self.count
 
  def __feature_len__(self):
    return self.input_len

def convert_Y(Y):
  uniques = list(set(Y))
  y_converted = np.copy(Y)
  for i in range(len(Y)):
    y_converted[i] = uniques.index(Y[i])
  return y_converted

## data loader and processor
def proc_data(X, Y, batch_size):
  X = torch.from_numpy(X).double()
  Y = convert_Y(Y)
  Y = torch.from_numpy(Y).type(torch.LongTensor)
  data = torch.utils.data.TensorDataset(X,Y)
  loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
  return loader

def load_train(dir, dataset_name, batch_size, downsample_rate):
  X, Y, n_sample, feature_len = read_ucr_downsample(dir+dataset_name+'/'+dataset_name+'_TRAIN', downsample_rate)
  train_loader = proc_data(X, Y, batch_size)
  return train_loader, X, Y, feature_len, len(set(Y))

def load_test(dir, dataset_name, downsample_rate):
  X, Y, n_test_sample, feature_len = read_ucr_downsample(dir+dataset_name+'/'+dataset_name+'_TEST', downsample_rate) 
  # ## pick a subset for testing
  # test_sample_idx = np.random.randint(0,3000,size=(100,))
  # test_sample_idx = range(100)
  # X = X[test_sample_idx,:,:]
  # Y = Y[test_sample_idx]
  test_loader = proc_data(X, Y, n_test_sample) # use full test data
  return test_loader, X, Y

def load_boosting_mse(dir, dataset_name, batch_size, downsample_rate):
  X, Y, n_sample, feature_len = read_ucr_downsample(dir+dataset_name+'/'+dataset_name+'_TRAIN', downsample_rate)
  X = torch.from_numpy(X).double()
  id = list(Y==1)
  data = torch.utils.data.TensorDataset(X,torch.zeros_like(X).view(X.shape[0],-1))
  train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
  X, Y, n_test_sample, feature_len = read_ucr_downsample(dir+dataset_name+'/'+dataset_name+'_TEST', downsample_rate) 
  X = torch.from_numpy(X).double()
  data = torch.utils.data.TensorDataset(X,torch.zeros_like(X).view(X.shape[0],-1))
  test_loader = torch.utils.data.DataLoader(data, batch_size=n_test_sample, shuffle=True)
  return train_loader, test_loader, X, Y, feature_len
 

## train and test
def train(epoch, batch_size, train_loader, model, optimizer, loss_func, **kwargs):
    model.train()
    train_loss = 0
    ft = 0
    bt = 0
    if 'scheduler' in kwargs: kwargs.get('scheduler').step()
    param_dict = kwargs.get('param_dict') if 'param_dict' in kwargs else None        
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data).float()
        if loss_func == loss_function_mse: label = Variable(label).type(torch.FloatTensor)
        else: label = Variable(label).type(torch.LongTensor)
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
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss*batch_size / len(train_loader.dataset)))
          if 'vis' in kwargs: 
            kwargs.get('vis').line(X=torch.FloatTensor([epoch]),Y=torch.FloatTensor([train_loss*batch_size/len(train_loader.dataset)]), win='loss_train', update='append' if epoch > 0 else None)
            if loss_func == loss_function_mse: kwargs.get('vis').line(Y=prediction[0], win='prediction')
    print('forward and backward time: ', ft, bt)

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
        loss = loss_func(label, prediction, model=model)
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

def train_hybrid(epoch, batch_size, train_loader, model, optimizer, loss_func=None, **kwargs):
    model.train()
    train_loss = 0
    ft = 0
    bt = 0
    if 'scheduler' in kwargs: kwargs.get('scheduler').step()
    param_dict = kwargs.get('param_dict') if 'param_dict' in kwargs else None        
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data).float()
        label = Variable(label).type(torch.LongTensor)
        optimizer.zero_grad()
        ts = time.time()
        signalreduce, prediction = model(data)
        ft += time.time() - ts
        loss = loss_function_hybrid(label, signalreduce, prediction)
        ts = time.time()
        loss.backward(retain_graph=True)
        bt += time.time() - ts
        #print('forward: ', time.time()-ts)
        optimizer.step()
        train_loss += loss.data.item()
        if batch_idx == len(train_loader.dataset)/batch_size-1:
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss*batch_size / len(train_loader.dataset)))
          if 'vis' in kwargs: 
            kwargs.get('vis').line(X=torch.FloatTensor([epoch]),Y=torch.FloatTensor([train_loss*batch_size/len(train_loader.dataset)]), win='loss_train', update='append' if epoch > 0 else None)
    print('forward and backward time: ', ft, bt)

def test_hybrid(epoch, test_loader, model, loss_func=None, **kwargs):
  if epoch % 10 == 0:
    model.eval()
    test_loss = 0
    test_auc = 0
    test_acc = 0
    correct = 0
    with torch.no_grad():
      for data, label in test_loader:
        data = Variable(data).float()
        label = Variable(label).type(torch.LongTensor)
        _, prediction = model(data)
        loss = loss_function_crossentropy(label, prediction)
        test_loss += loss.data.item()
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

def train_hybrid_switch(epoch, batch_size, train_loader, model, optimizer, loss_func=None, **kwargs):
    model.train()
    train_loss = 0
    ft = 0
    bt = 0
    if 'scheduler' in kwargs: kwargs.get('scheduler').step()
    param_dict = kwargs.get('param_dict') if 'param_dict' in kwargs else None        
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data).float()
        label = Variable(label).type(torch.LongTensor)
        optimizer.zero_grad()
        ts = time.time()
        if epoch < 200: interval = 2
        else: interval = 2
        if epoch % interval == 0:
          prediction = model(data, 'dtwnet')
          loss = loss_function_crossentropy(label, prediction)
        else:
          signalreduce = model(data, 'boosting')
          loss = loss_function_mse_flat(label, signalreduce)
        ft += time.time() - ts
        ts = time.time()
        loss.backward(retain_graph=True)
        bt += time.time() - ts
        #print('forward: ', time.time()-ts)
        optimizer.step()
        train_loss += loss.data.item()
        if batch_idx == len(train_loader.dataset)/batch_size-1:
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss*batch_size / len(train_loader.dataset)))
          if 'vis' in kwargs: 
            kwargs.get('vis').line(X=torch.FloatTensor([epoch]),Y=torch.FloatTensor([train_loss*batch_size/len(train_loader.dataset)]), win='loss_train', update='append' if epoch > 0 else None)
    print('forward and backward time: ', ft, bt)

def test_hybrid_switch(epoch, test_loader, model, loss_func=None, **kwargs):
  if epoch % 10 == 0:
    model.eval()
    test_loss = 0
    test_auc = 0
    test_acc = 0
    correct = 0
    with torch.no_grad():
      for data, label in test_loader:
        data = Variable(data).float()
        label = Variable(label).type(torch.LongTensor)
        prediction = model(data, 'dtwnet')
        loss = loss_function_crossentropy(label, prediction)
        test_loss += loss.data.item()
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


## test code
# def test_code():
#   model = Model.CONVDTW(10, 10, 10)
#   data = Variable(torch.randn(10))
#   labels = Variable(torch.randn(1))
# 
#   # model = CONVDTW_LAYER(5, 10, 100, 10)
#   # data = Variable(torch.randn(1, 100))
#   # labels = Variable(torch.randn(5, 91))
# 
#   optimizer = optim.SGD(model.parameters(), lr=1e-4)
#   for i in range(1000):
#     optimizer.zero_grad()
#     predictions = model(data)
#     #loss = loss_function_multi(5, labels, predictions)
#     loss = loss_function(labels, predictions)
#     loss.backward(retain_graph=True)
#     optimizer.step()
#     print(loss)
#     for name, param in model.named_parameters():
#       if param.requires_grad:
#         print(name, param.data)
#     print(data)


