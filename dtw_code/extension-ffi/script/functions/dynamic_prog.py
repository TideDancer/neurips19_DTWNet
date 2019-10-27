# functions/dynamic_prog.py
import torch
from torch.autograd import Function
from _ext import dynamic_prog_lib


class DynamicProgFunction(Function):
    def forward(self, input1, input2):
        # if not input1.is_cuda:
        #     my_lib.my_lib_add_forward(input1, input2, output)
        # else:
        #     my_lib.my_lib_add_forward_cuda(input1, input2, output)
        # path_num = 2
        path = torch.zeros((1, len(input1)+len(input2), 2), dtype=torch.int32)
        path_length = torch.zeros((1,), dtype=torch.int32)
        distances = torch.zeros(len(input2), len(input1), dtype=torch.float32)
        accumulated_cost = torch.zeros(len(input2), len(input1), dtype=torch.float32)
        start_pos = torch.zeros(len(input2), len(input1), dtype=torch.int32)
        
        # output = dynamic_prog_lib.dynamic_prog_lib_forward(input1, input2, distances, path)
        # output = dynamic_prog_lib.dynamic_prog_lib_forward_spring(input1, input2, path_num, path, path_length);
        # output = dynamic_prog_lib.dynamic_prog_lib_forward_spring(input1, input2, path_num, path, path_length, start_pos, accumulated_cost);

        eps = 0.1
        path_num = dynamic_prog_lib.dynamic_prog_lib_forward_spring_epspath(input1, input2, eps, path, path_length)
        # path_num = dynamic_prog_lib.dynamic_prog_lib_forward_spring_epspath(input1, input2, eps, path, path_length, accumulated_cost, start_pos) # for debug purpose

        # print input1
        # print input2
        print path
        print path_length
        # print start_pos.transpose(0,1)
        # print accumulated_cost.transpose(0,1)
        # return torch.Tensor([output])
        return torch.Tensor([path_num])

    def backward(self, grad_output, input):
        grad_input = grad_output.new()
        # if not grad_output.is_cuda:
        #     my_lib.my_lib_add_backward(grad_output, grad_input)
        # else:
        #     my_lib.my_lib_add_backward_cuda(grad_output, grad_input)
        dynamic_prog_lib.dynamic_prog_lib_backward(grad_output, input, grad_input)
        return grad_input
