import sys

## the argv[1] should be the filename without .py inside configs/ directory
## e.g. python main.py config_sample 0, will import configs/config_sample.py
config = sys.argv[1]
config = 'from configs.'+config+' import *'
exec(config)
print(model)

## the argv[2] controls whether to use visdom, 1 for enable
vis_enable = int(sys.argv[2])

## verbose to show parameters grads
verbose = int(sys.argv[3])

## draw sample data for illustration
if vis_enable == 1:
  vis = visdom.Visdom(env='model_1')
  vis.line(Y=np.transpose(sampleX[0:4,0,:]), opts=dict(legend=list(sampleY[0:4]),), win='data0')

## training loop
for epoch in range(500):
  if vis_enable == 1:
    for i in range(len(model.dtwlayers)):
      for j in range(n_filter):
        vis.line(Y=model.dtwlayers[i].filters[j].kernel.data, win='filter'+str(i)+str(j), opts=dict(title='filter'+str(i)+','+str(j),))

  start_time = time.time()
  
  ## train
  if vis_enable: train_func(epoch, batch_size, train_loader, model, optimizer, loss_func, scheduler=scheduler, vis=vis)
  else         : train_func(epoch, batch_size, train_loader, model, optimizer, loss_func, scheduler=scheduler)
  print('training time per epoch: ', time.time()-start_time)

  ## test
  if vis_enable: test_func(epoch, test_loader, model, loss_func, vis=vis)
  else         : test_func(epoch, test_loader, model, loss_func)

  ## print params
  if verbose:
    for name, param in model.named_parameters():
      if param.requires_grad:
        if 'kernel' in name: 
          # print(name, torch.sum(torch.abs((param.data))))
          print(name, torch.sum(torch.abs((param.grad))))
          # print(name, param.grad)

