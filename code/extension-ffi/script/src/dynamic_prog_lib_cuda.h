/* src/dynamic_prog.h */
int dynamic_prog_lib_forward_cuda(const THCudaTensor *input1, const THCudaTensor *input2, THCudaTensor* path);
// int dynamic_prog_lib_backward(THFloatTensor *grad_output, THFloatTensor *input, THFloatTensor *grad_input);
//THCudaTensor* dynamic_prog_lib_forward_spring_cuda(THCudaTensor *input1, THCudaTensor *input2, int path_num, THCudaTensor *path, THCudaTensor *path_length);
// THFloatTensor* dynamic_prog_lib_forward_spring(THFloatTensor *input1, THFloatTensor *input2, int path_num, THIntTensor *path, THIntTensor *path_length, THIntTensor *start_pos, THFloatTensor *accumulated_cost);// for debug purpose

// int dynamic_prog_lib_forward_spring_epspath_cuda(THCudaTensor *input1, THCudaTensor *input2, float eps, THCudaTensor *path, THCudaTensor *path_length);
// int dynamic_prog_lib_forward_spring_epspath(THFloatTensor *input1, THFloatTensor *input2, float eps, THIntTensor *path, THIntTensor *path_length, THFloatTensor* accumulated_cost, THIntTensor* start_pos); // for debug purpose

