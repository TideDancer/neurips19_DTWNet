/* src/dynamic_prog.h */
int dynamic_prog_lib_forward(const THFloatTensor *input1, const THFloatTensor *input2, THIntTensor* path);
//int dynamic_prog_lib_backward(THFloatTensor *grad_output, THFloatTensor *input, THFloatTensor *grad_input);
THFloatTensor* dynamic_prog_lib_forward_spring(THFloatTensor *input1, THFloatTensor *input2, int path_num, THIntTensor *path, THIntTensor *path_length);
// THFloatTensor* dynamic_prog_lib_forward_spring(THFloatTensor *input1, THFloatTensor *input2, int path_num, THIntTensor *path, THIntTensor *path_length, THIntTensor *start_pos, THFloatTensor *accumulated_cost);// for debug purpose
//

int dynamic_prog_lib_forward_spring_epspath(THFloatTensor *input1, THFloatTensor *input2, float eps, THIntTensor *path, THIntTensor *path_length);
// int dynamic_prog_lib_forward_spring_epspath(THFloatTensor *input1, THFloatTensor *input2, float eps, THIntTensor *path, THIntTensor *path_length, THFloatTensor* accumulated_cost, THIntTensor* start_pos); // for debug purpose

