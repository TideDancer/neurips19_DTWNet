import numpy as np
import torch
import dtwco
from torch import nn
from _ext import dynamic_prog_lib

## MLP

class MLP(nn.Module):
  def __init__(self, _n_layer, _input_len, _output_len):
    super(MLP, self).__init__()
    self.n_layer = _n_layer
    tmp = [nn.Linear(_input_len, 128)]
    for i in range(1, _n_layer-1):
      tmp.append(nn.Linear(128, 128))
    self.linears = nn.ModuleList(tmp)
    self.logits = nn.Linear(128, _output_len)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()
    self.dropout = nn.Dropout()

  def forward(self, input):
    t = input.view(input.shape[0],-1)
    for i in range(self.n_layer-1):
      t = self.linears[i](t)
      # t = self.dropout(t)
      t = self.relu(t)
    t = self.logits(t)
    return self.softmax(t)


## DTW operation classes

class DTW(nn.Module):
  def __init__(self, _kernel_len, _input_len, **kwargs):
    super(DTW, self).__init__()
    self.kernel_len = _kernel_len
    self.input_len = _input_len
    if '_kernel' not in kwargs:
      self.kernel = nn.Parameter(torch.randn(_kernel_len))
    else:
      self.kernel = nn.Parameter(kwargs.get('_kernel'))
  def compute(self, input):
    pass
  def forward(self, input):
    return self.compute(input)

   
class DTW_FULL(DTW):
  def __init__(self, _kernel_len, _input_len, **kwargs):
    super(DTW_FULL, self).__init__(_kernel_len, _input_len, **kwargs)
    self.out_len = 1
 
  def compute(self, input):
    # ## debugging code
    # distances = torch.zeros((self.input_len, self.kernel_len), dtype=torch.float32)
    # dummy = torch.zeros((self.input_len, self.kernel_len), dtype=torch.float32)
    # for i in range(self.input_len):
    #     for j in range(self.kernel_len):
    #         distances[i,j] = (self.kernel[j]-y[i])**2  

    path = torch.zeros((self.input_len+self.kernel_len, 2), dtype=torch.int32)
    path_len = dynamic_prog_lib.dynamic_prog_lib_forward(self.kernel, input, path)
    return (sum([(self.kernel[path[i][0]]-input[path[i][1]])**2 for i in range(path_len)]))


class DTW_FULL_DTWCO(DTW):
  def __init__(self, _kernel_len, _input_len, **kwargs):
    super(DTW_FULL_DTWCO, self).__init__(_kernel_len, _input_len, **kwargs)
    self.out_len = 1
 
  def compute(self, input):
    dist, cost, path = dtwco.dtw(self.kernel.detach().numpy(), input.detach().numpy(), metric='sqeuclidean', dist_only=False)
    return sum([(self.kernel[path[0][i]]-input[path[1][i]])**2 for i in range(len(path[0]))])


class DTW_SPRING_EPS(DTW):
  def __init__(self, _kernel_len, _input_len, _eps=0.5, **kwargs):
    super(DTW_SPRING_EPS, self).__init__(_kernel_len, _input_len, **kwargs)
    self.eps = _eps
    self.out_len = 1
    self.max_path = 30

  def compute(self, input):
    path = torch.zeros((self.max_path, self.input_len+self.kernel_len, 2), dtype=torch.int32)
    path_length = torch.zeros((self.max_path,), dtype=torch.int32)
    path_num = dynamic_prog_lib.dynamic_prog_lib_forward_spring_epspath(self.kernel, input, self.eps, path, path_length);
    ## random choose one path and return
    k = np.random.random_integers(0,path_num-1) if path_num > 0 else 0
    return sum([(self.kernel[path[k][i][0]]-input[path[k][i][1]])**2 for i in range(path_length[k])])
    # ## sum all path and return
    # return sum([sum([(self.kernel[path[k][i][0]]-input[path[k][i][1]])**2 for i in range(path_length[k])]) for k in range(path_num)])
  

class DTW_SPRING_NPATH(DTW):
  def __init__(self, _kernel_len, _input_len, _path_num=10, _eps=0.5, **kwargs):
    super(DTW_SPRING_NPATH, self).__init__(_kernel_len, _input_len, **kwargs)
    self.path_num = _path_num
    self.out_len = _path_num
    self.max_path = 30
    self.eps = 0.5

  def compute(self, input):
    path = torch.zeros((self.max_path, self.input_len+self.kernel_len, 2), dtype=torch.int32)
    path_length = torch.zeros((self.max_path,), dtype=torch.int32)
    path_num = dynamic_prog_lib.dynamic_prog_lib_forward_spring_epspath(self.kernel, input, self.eps, path, path_length);
    # return sum([sum([(self.kernel[path[k][i][0]]-input[path[k][i][1]])**2 for i in range(path_length[k])]) for k in range(path_num)])
    out = torch.zeros((self.path_num,))
    for k in range(min(path_num, self.path_num)):
      out[k] = sum([(self.kernel[path[k][i][0]]-input[path[k][i][1]])**2 for i in range(path_length[k])])
    return out


class DTW_SPRING_ROW(DTW):
  def __init__(self, _kernel_len, _input_len, _path_num=10, _eps=0.5, **kwargs):
    super(DTW_SPRING_ROW, self).__init__(_kernel_len, _input_len, **kwargs)
    self.path_num = _path_num
    self.out_len = _input_len
    self.max_path = 30
    self.eps = 0.5

  def compute(self, input):
    path = torch.zeros((self.max_path, self.input_len+self.kernel_len, 2), dtype=torch.int32)
    path_length = torch.zeros((self.max_path,), dtype=torch.int32)
    path_num = dynamic_prog_lib.dynamic_prog_lib_forward_spring_epspath(self.kernel, input, self.eps, path, path_length);
    out = torch.zeros((self.out_len,))
    for k in range(path_num):
      end = path[k][0][1]
      start = path[k][path_length[k]-1][1]
      # print(start,end)
      out[start:end] = sum([(self.kernel[path[k][i][0]]-y[path[k][i][1]])**2 for i in range(path_length[k])])
    return out


## python models for row output

class pythonDTW_FULL_ROW(DTW):

  def __init__(self, _kernel_len, _input_len):
    super(pythonDTW_FULL_ROW, self).__init__(_kernel_len, _input_len)
    self.out_len = _input_len

  def compute(self, input):
    accumulated_cost = torch.zeros(self.input_len, self.kernel_len)
    distances = torch.zeros(self.input_len, self.kernel_len)

    for i in range(self.input_len):
        for j in range(self.kernel_len):
            distances[i,j] = (self.kernel[j]-input[i])**2  
    accumulated_cost[0,0] = distances[0,0]
 
    for i in range(1, self.kernel_len):
        accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]   
    for i in range(1, self.input_len):
        accumulated_cost[i,0] = distances[i,0] + accumulated_cost[i-1, 0]   
  
    for i in range(1, self.input_len):
      for j in range(1, self.kernel_len):
          accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]

    return accumulated_cost[:, self.kernel_len-1]
    
    # path = [[self.kernel_len-1, self.input_len-1]]
    # i = self.input_len-1
    # j = self.kernel_len-1
    # while i>0 and j>0:
    #     if i==0:
    #         j = j - 1
    #     elif j==0:
    #         i = i - 1
    #     else:
    #         if self.accumulated_cost[i-1, j] == min(self.accumulated_cost[i-1, j-1], self.accumulated_cost[i-1, j], self.accumulated_cost[i, j-1]):
    #             i = i - 1
    #         elif self.accumulated_cost[i, j-1] == min(self.accumulated_cost[i-1, j-1], self.accumulated_cost[i-1, j], self.accumulated_cost[i, j-1]):
    #             j = j - 1
    #         else:
    #             i = i - 1
    #             j = j - 1
    #     if i < self.input_len-1 and j < self.kernel_len-1:
    #         path.append([j, i])
    # #path.append([0,0])
    # return (sum([(self.kernel[path[i][0]]-y[path[i][1]])**2 for i in range(len(path))]))


class pythonDTW_SPRING_ROW(DTW):

  def __init__(self, _kernel_len, _input_len):
    super(pythonDTW_SPRING_ROW, self).__init__(_kernel_len, _input_len)
    self.out_len = _input_len

  def compute(self, input):
    accumulated_cost = torch.zeros(self.input_len, self.kernel_len)
    distances = torch.zeros(self.input_len, self.kernel_len)
    # start_pos = (torch.zeros(self.subseq_len, self.kernel_len))

    for i in range(self.input_len):
        for j in range(self.kernel_len):
            distances[i,j] = (self.kernel[j]-input[i])**2  
    accumulated_cost[0,0] = distances[0,0]
  
    for i in range(1, self.kernel_len):
        accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]   
        # start_pos[0,i] = 0
    for i in range(1, self.input_len):
        accumulated_cost[i,0] = distances[i,0]
        # start_pos[i,0] = i
  
    for i in range(1, self.input_len):
      for j in range(1, self.kernel_len):
          accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]

    return accumulated_cost[:, self.kernel_len-1]


## DTW layers

class DTWLAYER(nn.Module):
  def __init__(self, _dtw_list):
    super(DTWLAYER, self).__init__()
    self.num_filter = len(_dtw_list)
    self.filters = nn.ModuleList(_dtw_list)
    self.filter_outlen = _dtw_list[0].out_len
    self.out_len = self.num_filter*self.filter_outlen
  def forward_one_batch(self, input):
    out = torch.zeros((self.num_filter, self.filter_outlen))
    for i in range(self.num_filter):
      out[i] = self.filters[i].forward(input) 
    return out
  def forward(self, input):
    out = torch.tensor([]).new_empty(input.shape[0], self.num_filter, self.filter_outlen)
    for k in range(input.shape[0]): # batch size
      out[k] = self.forward_one_batch(input[k])
    return out

class DTWLAYER_SHORTKERNEL(DTWLAYER):
  def __init__(self, _dtw_list, _input_chan, _input_len, _proc_len=20, _step_len=5):
    super(DTWLAYER_SHORTKERNEL, self).__init__(_dtw_list)
    self.input_chan = _input_chan
    self.input_len = _input_len
    self.out_chan = self.num_filter*self.input_chan
    self.kernel_len = _dtw_list[0].kernel_len
    self.proc_len = _proc_len #self.kernel_len*2 # process length of subsequence in the input
    self.step_len = _step_len #int(self.kernel_len/2)   # stride
    self.chan_outlen = int((_input_len - self.proc_len + 1)/self.step_len)
    self.out_len = self.out_chan*self.chan_outlen
  def forward_one_batch(self, input):
    input = input.view(self.input_chan, -1)
    out = torch.zeros((self.out_chan, self.chan_outlen))
    t = 0
    for k in range(self.input_chan):
      for i in range(self.num_filter):
        j = 0
        p = 0
        while j < input.shape[1] - self.proc_len - self.step_len + 1:
          out[t,p] = self.filters[i].forward(input[k, j:j+self.proc_len])
          j += self.step_len
          p += 1
        t += 1
    return out
  def forward(self, input): # input: batch_size * input_chan * len
    out = torch.zeros(input.shape[0], self.out_chan, self.chan_outlen)
    for k in range(input.shape[0]): # batch size
      out[k] = self.forward_one_batch(input[k])
    return out

## DTW networks

class DTWNET_BASE(nn.Module):
  def __init__(self, _dtwlayer_list):
    super(DTWNET_BASE, self).__init__()
    self.num_dtwlayer = len(_dtwlayer_list)
    self.dtwlayers = nn.ModuleList(_dtwlayer_list)
    self.dtwlayer_outlen = _dtwlayer_list[-1].out_len
  def construct_dtw(self, input):
    pass
  def forward(self, input):
    pass


class DTWNET_MSE(DTWNET_BASE):
  def __init__(self, _dtwlayer_list):
    super(DTWNET_MSE, self).__init__(_dtwlayer_list)
  def forward(self, input):
    t = self.dtwlayers[-1](input.view(input.shape[0], -1))
    return t.view(input.shape[0], -1)


class DTWNET(DTWNET_BASE):
  def __init__(self, _n_class, _dtwlayer_list):
    super(DTWNET, self).__init__(_dtwlayer_list)
    self.mlps = nn.ModuleList([MLP(3, self.dtwlayer_outlen, _n_class)])

  def construct_dtw(self, input):
    pass

  def forward(self, input):
    t = self.construct_dtw(input)
    t = t.view(input.shape[0], -1)
    return self.mlps[0](t)


class DTWNET_SINGLE(DTWNET):
  def __init__(self, _n_class, _dtwlayer_list):
    super(DTWNET_SINGLE, self).__init__(_n_class, _dtwlayer_list)

  def construct_dtw(self, input):
    t = self.dtwlayers[0](input.view(input.shape[0], -1))
    for i in range(1,len(self.dtwlayers)):
      t = self.dtwlayers[i](t)
    t = t.view(input.shape[0], -1)
    return t

## CNN

class CONV_MLP(nn.Module):
  def __init__(self, _n_class, _input_len, _n_filter, _kernel_len):
    super(CONV_MLP, self).__init__()
    self.n_class = _n_class
    self.kernel_size = _kernel_len
    self.conv = nn.Conv1d(1, _n_filter, kernel_size=_kernel_len, stride=1, padding=0, bias=True )
    self.mlps = nn.ModuleList([MLP(3, _n_filter*(_input_len-self.kernel_size+1), _n_class)])
    self.dropout = nn.Dropout()
    #self.bn = nn.BatchNorm1d(_input_len-self.kernel_size+1)
  def forward(self, input):
    t = self.conv(input)
    t = t.view(t.shape[0], -1)
    t = self.dropout(t)
    return self.mlps[0](t)


## boosting model

class DTW_BOOSTING(DTW):  # return both input_reduce and dtw_path
  def __init__(self, _kernel_len, _input_len, weight, **kwargs):
    super(DTW_BOOSTING, self).__init__(_kernel_len, _input_len, **kwargs)
    self.out_len = _input_len
    self.weight = weight
 
  def compute(self, input):
    dist, cost, path = dtwco.dtw(self.kernel.detach().numpy(), input.detach().numpy(), dist_only=False)
    out = torch.Tensor.clone(input)
    for i in range(len(path[0])):
      out[path[1][i]] -= self.kernel[path[0][i]]
    out *= self.weight
    return out, sum([(self.kernel[path[0][i]]-input[path[1][i]])**2 for i in range(len(path[0]))])


class DTWLAYER_BOOSTING(DTWLAYER):

  def __init__(self, _dtw_list):
    super(DTWLAYER_BOOSTING, self).__init__(_dtw_list)
    self.out_len = self.filter_outlen
 
  def forward_one_batch(self, input):
    out = torch.zeros(self.out_len)
    path_sum = torch.tensor([]).new_empty(self.num_filter)
    for i in range(self.num_filter):
      tmp, path_sum[i] = self.filters[i].forward(input) 
      out += tmp
    return out, path_sum

  def forward(self, input):
    out = torch.tensor([]).new_empty(input.shape[0], self.filter_outlen)
    path_sum = torch.tensor([]).new_empty(input.shape[0], self.num_filter)
    for k in range(input.shape[0]): # batch size
      out[k], path_sum[k] = self.forward_one_batch(input[k])
    return out, path_sum


class DTWNET_BOOSTING(DTWNET_BASE):
  def __init__(self, _n_class, _dtwlayer_list):
    super(DTWNET_BOOSTING, self).__init__(_dtwlayer_list)

  def construct_dtw(self, input):
    t = input.view(input.shape[0], -1)
    for i in range(self.num_dtwlayer):
      t, _ = self.dtwlayers[i](t)
    t = t.view(input.shape[0], -1)
    return t
 
  def construct_dtw_dual(self, input):
    t = input.view(input.shape[0], -1)
    t2 = torch.tensor([]).new_empty(input.shape[0], self.num_dtwlayer, self.dtwlayers[-1].num_filter)
    for i in range(self.num_dtwlayer):
      t, t2[:,i,:] = self.dtwlayers[i](t)
    t = t.view(input.shape[0], -1)
    t2 = t2.view(input.shape[0], -1)
    return t, t2

  def forward(self, input, **kwargs):
    if 'dual' in kwargs:
      return self.construct_dtw_dual(input)
    else: 
      return self.construct_dtw(input)


# ## hybrid model
# 
# class HYBRID(DTWNET_BOOSTING):
#   def __init__(self, _n_class, _dtwlayer_list):
#     super(HYBRID, self).__init__(_n_class, _dtwlayer_list)
#     self.mlps = nn.ModuleList([MLP(3, self.num_dtwlayer*self.dtwlayers[-1].num_filter, _n_class)])
# 
#   def construct_dtw(self, input):
#     t = input.view(input.shape[0], -1)
#     t2 = torch.tensor([]).new_empty(input.shape[0], self.num_dtwlayer, self.dtwlayers[-1].num_filter)
#     for i in range(self.num_dtwlayer):
#       t, t2[:,i,:] = self.dtwlayers[i](t)
#     t = t.view(input.shape[0], -1)
#     t2 = t2.view(input.shape[0], -1)
#     return t, t2
#   
#   def forward(self, input):
#     signal_reduce, pathsum = self.construct_dtw(input)
#     pathsum = self.mlps[0].forward(pathsum)
#     return signal_reduce, pathsum
# 
# 
# class HYBRID_SWITCH(nn.Module):
#   def __init__(self, _boosting_list, _dtwnet_list):
#     super(HYBRID_SWITCH, self).__init__()
#     self.boostings = nn.ModuleList(_boosting_list)
#     self.dtwnets = nn.ModuleList(_dtwnet_list)
# 
#   def forward(self, input, switch):
#     if switch == 'boosting': return self.boostings[-1].forward(input)
#     if switch == 'dtwnet': return self.dtwnets[-1].forward(input)
#     else: print('WARNING: You must specify switch to be "boosting" or "dtwnet"')
