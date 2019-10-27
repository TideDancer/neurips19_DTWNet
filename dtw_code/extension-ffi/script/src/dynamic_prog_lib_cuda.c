// to support python 3 where all ints are long
#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong
#endif

#include <TH/THC.h>
#include <math.h>

#define INF 999999

int dynamic_prog_lib_forward_cuda(const THCudaTensor *input1, const THCudaTensor *input2, THCudaTensor* path)
// input1: kernel, must be tensor of size (k,) where k is the length
// input2: subseq, must be tensor of size (s,) where s is the length
{
    int i, j, path_size;
    double tmp;
    int len2 = THCudaTensor_size(input2, 0);
    int len1 = THCudaTensor_size(input1, 0);

    // ---------------------- compute dtw -------------------

    THCudaTensor* accumulated_cost = THCudaTensor_newWithSize2d(len2, len1);
    THCudaTensor* distances = THCudaTensor_newWithSize2d(len2, len1);

    for(i=0; i<len2; i++)
        for(j=0; j<len1; j++){
            tmp = (THCudaTensor_get1d(input1, j) - THCudaTensor_get1d(input2, i))*(THCudaTensor_get1d(input1, j) - THCudaTensor_get1d(input2, i));
            THCudaTensor_set2d(distances, i, j, tmp);
    }
    THCudaTensor_set2d(accumulated_cost, 0, 0, THCudaTensor_get2d(distances, 0, 0));
 
		for(i=1; i<len1; i++){
        tmp = THCudaTensor_get2d(distances, 0, i) + THCudaTensor_get2d(accumulated_cost, 0, i-1);
        THCudaTensor_set2d(accumulated_cost, 0, i, tmp);  
    }
		for(i=1; i<len2; i++){
        tmp = THCudaTensor_get2d(distances, i, 0) + THCudaTensor_get2d(accumulated_cost, i-1, 0);
        THCudaTensor_set2d(accumulated_cost, i, 0, tmp);
		}
  
    for(i=1; i<len2; i++)
				for(j=1; j<len1; j++){
            tmp = fmin(THCudaTensor_get2d(accumulated_cost, i-1, j-1), THCudaTensor_get2d(accumulated_cost, i-1, j));
            tmp = fmin(tmp, THCudaTensor_get2d(accumulated_cost, i, j-1));
            tmp = tmp + THCudaTensor_get2d(distances, i, j);
            THCudaTensor_set2d(accumulated_cost, i, j, tmp);
		}

    // ---------------------- obtain path -------------------
    path_size = 0;
    i = len2-1;
    j = len1-1;

    THCudaTensor_set2d(path, 0, 0, j);
    THCudaTensor_set2d(path, 0, 1, i);
    path_size ++;

    while(i>0 || j>0){
        if(i==0)
            j = j - 1;
        else if(j==0)
            i = i - 1;
				else{
            tmp = fmin(THCudaTensor_get2d(accumulated_cost, i-1, j-1), THCudaTensor_get2d(accumulated_cost, i-1, j));
            tmp = fmin(tmp, THCudaTensor_get2d(accumulated_cost, i, j-1));
            if(tmp == THCudaTensor_get2d(accumulated_cost, i-1, j)){
                i = i - 1;
            }
						else if(tmp == THCudaTensor_get2d(accumulated_cost, i, j-1)){ 
                j = j - 1; 
            }
						else{
                i = i - 1;
                j = j - 1;
						}
				}
        THCudaTensor_set2d(path, path_size, 0, j);
        THCudaTensor_set2d(path, path_size, 1, i);
        path_size ++;
    }

    THCudaTensor_free(accumulated_cost);
    THCudaTensor_free(distances);
    return path_size;
}

 
// THCudaTensor* dynamic_prog_lib_forward_spring(THCudaTensor *input1, THCudaTensor *input2, int path_num, THCudaTensor *path, THCudaTensor *path_length)
// //THCudaTensor* dynamic_prog_lib_forward_spring(THCudaTensor *input1, THCudaTensor *input2, int path_num, THCudaTensor *path, THCudaTensor *path_length, THCudaTensor *start_pos, THCudaTensor *accumulated_cost) // for debug purpose
// // input1: kernel, must be tensor of size (k,) where k is the length
// // input2: subseq, must be tensor of size (s,) where s is the length
// {
//     int i, j, k, pos, end, diag_pos, left_pos, down_pos, path_size;
//     double tmp, dtw_result, diag, left, down;
//     int len2 = THCudaTensor_size(input2, 0);
//     int len1 = THCudaTensor_size(input1, 0);
// 
//     // ---------------------- compute dtw -------------------
// 
//     THCudaTensor* accumulated_cost = THCudaTensor_newWithSize2d(len2, len1);
//     THCudaTensor* distances = THCudaTensor_newWithSize2d(len2, len1);
//     THCudaTensor* start_pos = THCudaTensor_newWithSize2d(len2, len1);
// 
//     for(i=0; i<len2; i++)
//         for(j=0; j<len1; j++){
//             tmp = (THCudaTensor_get1d(input1, j) - THCudaTensor_get1d(input2, i))*(THCudaTensor_get1d(input1, j) - THCudaTensor_get1d(input2, i));
//             THCudaTensor_set2d(distances, i, j, tmp);
//     }
//     THCudaTensor_set2d(accumulated_cost, 0, 0, THCudaTensor_get2d(distances, 0, 0));
//     THCudaTensor_set2d(start_pos, 0, 0, 0);
//  
// 		for(i=1; i<len1; i++){
//         tmp = THCudaTensor_get2d(distances, 0, i) + THCudaTensor_get2d(accumulated_cost, 0, i-1);
//         THCudaTensor_set2d(accumulated_cost, 0, i, tmp);  
//         THCudaTensor_set2d(start_pos, 0, i, 0);
//     }
// 		for(i=1; i<len2; i++){
//         tmp = THCudaTensor_get2d(distances, i, 0);
//         THCudaTensor_set2d(accumulated_cost, i, 0, tmp);
//         THCudaTensor_set2d(start_pos, i, 0, i);
// 		}
//   
//     for(i=1; i<len2; i++)
// 				for(j=1; j<len1; j++){
//             // compute distance matrix
//             diag = THCudaTensor_get2d(accumulated_cost, i-1, j-1);
//             left = THCudaTensor_get2d(accumulated_cost, i-1, j);
//             down = THCudaTensor_get2d(accumulated_cost, i, j-1);
//             tmp = fmin(fmin(diag, left), down);
//             THCudaTensor_set2d(accumulated_cost, i, j, tmp+THCudaTensor_get2d(distances, i, j));
// 
//             // compute augmented starting_position matrix
//             diag_pos = THCudaTensor_get2d(start_pos, i-1, j-1);
//             left_pos = THCudaTensor_get2d(start_pos, i-1, j);
//             down_pos = THCudaTensor_get2d(start_pos, i, j-1);
//             if(tmp==diag) pos = diag_pos;
//             else if(tmp==left) pos = left_pos;
//             else pos = down_pos;
//             THCudaTensor_set2d(start_pos, i, j, pos);
// 		}
// 		
//     // ---------------------- obtain path -------------------
//     THCudaTensor* modify_array = THCudaTensor_newWithSize1d(len2);
//     THCudaTensor* dtw_results = THCudaTensor_newWithSize1d(path_num);
//     THCudaTensor* start_array = THCudaTensor_newWithSize1d(path_num);
//     THCudaTensor* end_array   = THCudaTensor_newWithSize1d(path_num);
// 
// 		for(i=0; i<len2; i++){
//         THCudaTensor_set1d(modify_array, i, THCudaTensor_get2d(accumulated_cost, i, len1-1));
// 		}
// 		for(k=0; k<path_num; k++){
//         float minval = INF;
//         for(i=0; i<len2; i++){
//             tmp = THCudaTensor_get1d(modify_array, i);
// 						if(tmp < minval){
//                 minval = tmp;
//                 pos = THCudaTensor_get2d(start_pos, i, len1-1);
//                 end = i;
// 						}
// 		    }
// 				for(i=0; i<len2; i++){
// 						if(pos == THCudaTensor_get2d(start_pos, i, len1-1)){
//                 THCudaTensor_set1d(modify_array, i, INF);
// 						}
// 				}
//         THCudaTensor_set1d(dtw_results, k, minval);
//         THCudaTensor_set1d(start_array, k, pos);
//         THCudaTensor_set1d(end_array,   k, end);
// 		}
// 		
//     for(k=0; k<path_num; k++){
// 
//         path_size = 0;
//         i = THCudaTensor_get1d(end_array, k);//len2-1;
//         j = len1-1;
//         pos = THCudaTensor_get1d(start_array, k);
// 
//         THCudaTensor_set3d(path, k, 0, 0, j);
//         THCudaTensor_set3d(path, k, 0, 1, i);
//         path_size ++;
// 
//         while(i>0 || j>0){
//             if(i==0)
//                 j = j - 1;
//             else if(j==0)
//                 break; //i = i - 1;
// 		    		else{
//                 // confirm starting pos matches then get left or right
//                 if(THCudaTensor_get2d(start_pos, i-1, j-1) == pos){
//                     diag = THCudaTensor_get2d(accumulated_cost, i-1, j-1);
//                 }
// 								else{
//                     diag = INF;
// 								}
// 								if(THCudaTensor_get2d(start_pos, i-1, j) == pos){
//                     left = THCudaTensor_get2d(accumulated_cost, i-1, j);
// 								}
// 								else{
//                     left = INF;
// 								}
//                 if(THCudaTensor_get2d(start_pos, i, j-1) == pos){
//                     down = THCudaTensor_get2d(accumulated_cost, i, j-1);
// 								}
// 								else{
//                     down = INF;
// 								}
//                 tmp = fmin(fmin(diag, left), down);
// 
//                 // obtain path
//                 if(tmp == left){ i = i-1; }
//                 else if(tmp == down){ j = j-1; }
//                 else { i = i-1; j = j-1; }
//  
// 		    		}
//             THCudaTensor_set3d(path, k, path_size, 0, j);
//             THCudaTensor_set3d(path, k, path_size, 1, i);
//             path_size ++;
//         }
//         
//         THCudaTensor_set1d(path_length, k, path_size);
// 
//         //for(i=0; i<path_size; i++){printf("%d, %d, %d\n", i, THCudaTensor_get2d(path, i, 0), THCudaTensor_get2d(path, i, 1));}
//     }
// 
//     THCudaTensor_free(accumulated_cost);
//     THCudaTensor_free(distances);
//     THCudaTensor_free(start_pos);
//     THCudaTensor_free(modify_array);
//     THCudaTensor_free(dtw_results);
//     THCudaTensor_free(start_array);
//     THCudaTensor_free(end_array)  ;
// 
//     return dtw_results;
// }



// int dynamic_prog_lib_forward_spring_epspath(THCudaTensor *input1, THCudaTensor *input2, float eps, THCudaTensor *path, THCudaTensor *path_length)
// //int dynamic_prog_lib_forward_spring_epspath(THCudaTensor *input1, THCudaTensor *input2, float eps, THCudaTensor *path, THCudaTensor *path_length, THCudaTensor* accumulated_cost, THCudaTensor* start_pos)
// // input1: kernel, must be tensor of size (k,) where k is the length
// // input2: subseq, must be tensor of size (s,) where s is the length
// {
//     int i, j, k, pos, end, diag_pos, left_pos, down_pos, path_size;
//     double tmp, dtw_result, diag, left, down;
//     int len2 = THCudaTensor_size(input2, 0);
//     int len1 = THCudaTensor_size(input1, 0);
// 
//     // ---------------------- compute dtw -------------------
// 
//     THCudaTensor* distances = THCudaTensor_newWithSize2d(len2, len1);
//     THCudaTensor* accumulated_cost = THCudaTensor_newWithSize2d(len2, len1);
//     THCudaTensor* start_pos = THCudaTensor_newWithSize2d(len2, len1);
// 
//     for(i=0; i<len2; i++)
//         for(j=0; j<len1; j++){
//             tmp = (THCudaTensor_get1d(input1, j) - THCudaTensor_get1d(input2, i))*(THCudaTensor_get1d(input1, j) - THCudaTensor_get1d(input2, i));
//             THCudaTensor_set2d(distances, i, j, tmp);
//     }
//     THCudaTensor_set2d(accumulated_cost, 0, 0, THCudaTensor_get2d(distances, 0, 0));
//     THCudaTensor_set2d(start_pos, 0, 0, 0);
//  
// 		for(i=1; i<len1; i++){
//         tmp = THCudaTensor_get2d(distances, 0, i) + THCudaTensor_get2d(accumulated_cost, 0, i-1);
//         THCudaTensor_set2d(accumulated_cost, 0, i, tmp);  
//         THCudaTensor_set2d(start_pos, 0, i, 0);
//     }
// 		for(i=1; i<len2; i++){
//         tmp = THCudaTensor_get2d(distances, i, 0);
//         THCudaTensor_set2d(accumulated_cost, i, 0, tmp);
//         THCudaTensor_set2d(start_pos, i, 0, i);
// 		}
//   
//     for(i=1; i<len2; i++)
// 				for(j=1; j<len1; j++){
//             // compute distance matrix
//             diag = THCudaTensor_get2d(accumulated_cost, i-1, j-1);
//             left = THCudaTensor_get2d(accumulated_cost, i-1, j);
//             down = THCudaTensor_get2d(accumulated_cost, i, j-1);
//             tmp = fmin(fmin(diag, left), down);
//             THCudaTensor_set2d(accumulated_cost, i, j, tmp+THCudaTensor_get2d(distances, i, j));
// 
//             // compute augmented starting_position matrix
//             diag_pos = THCudaTensor_get2d(start_pos, i-1, j-1);
//             left_pos = THCudaTensor_get2d(start_pos, i-1, j);
//             down_pos = THCudaTensor_get2d(start_pos, i, j-1);
//             if(tmp==diag) pos = diag_pos;
//             else if(tmp==left) pos = left_pos;
//             else pos = down_pos;
//             THCudaTensor_set2d(start_pos, i, j, pos);
// 		}
// 		
//     // ---------------------- obtain path -------------------
//     float thres; int path_num = 1; int effective_path_num = 0;
//     float tmp1; int pos1; int flag=0;
// 
//     // search for the minimum value and number of path
//     thres = THCudaTensor_get2d(accumulated_cost, 0, len1-1);
//     pos = THCudaTensor_get2d(start_pos, 0, len1-1);
// 		for(i=1; i<len2; i++){
//         tmp = THCudaTensor_get2d(accumulated_cost, i, len1-1);
//         pos1 = THCudaTensor_get2d(start_pos, i, len1-1);
// 				if(tmp<thres){ // obtain minimum value
//             thres = tmp;
// 				}
// 				if(pos1!=pos){ // count number of path
//             path_num ++;
//             pos = pos1;
// 				}
// 		}
//     thres = thres * (1+eps);
// 
//     THCudaTensor* start_array = THCudaTensor_newWithSize1d(path_num);
//     THCudaTensor* end_array   = THCudaTensor_newWithSize1d(path_num);
// 
//     // obtain each path
//     flag = 0;
//     tmp = THCudaTensor_get2d(accumulated_cost, 0, len1-1);
//     pos = THCudaTensor_get2d(start_pos, 0, len1-1);
//     end = 0; 
//     effective_path_num = 0; i = 1;
// 		while( i<len2 ){
//         if(tmp < thres) flag = 1;
//         tmp1 = THCudaTensor_get2d(accumulated_cost, i, len1-1);
//         pos1 = THCudaTensor_get2d(start_pos, i, len1-1);
//         
// 				if(tmp1 < thres && tmp1 < tmp && pos1 == pos){
//             tmp = tmp1;
//             end = i;
//             flag = 1;
// 				}
// 				else if(pos1 != pos && flag == 1){
//             // save
//             flag = 0;
//             //THCudaTensor_set1d(dtw_results, effective_path_num, tmp);
//             THCudaTensor_set1d(start_array, effective_path_num, pos);
//             THCudaTensor_set1d(end_array,   effective_path_num, end);
//             effective_path_num ++;
//             // renew
//             tmp = tmp1;
//             pos = pos1;
//             end = i;
// 				}
// 				else if(pos1 != pos && flag == 0){
//             // renew
//             tmp = tmp1;
//             pos = pos1;
//             end = i;
//             if(tmp < thres) flag = 1;
// 				}
//         i++;
// 		}
// 		if(flag == 1){ // handle the possible last effective path 
//         THCudaTensor_set1d(start_array, effective_path_num, pos);
//         THCudaTensor_set1d(end_array,   effective_path_num, end);
//         effective_path_num ++;
// 		}
// 
// 		// if(effective_path_num==0){
//     //     printf("%f,%f\n",thres,thres/(1+eps));
//     //     for(i=0;i<len2;i++) printf("%f,", THCudaTensor_get2d(accumulated_cost,i,len1-1));
//     //     printf("\n");
//     //     for(i=0;i<len2;i++) printf("%d,", THCudaTensor_get2d(start_pos,i,len1-1));
// 		// }
// 
//     // resize the path and path_length matrix
//     THCudaTensor_resize3d(path, effective_path_num, len1+len2, 2);
//     THCudaTensor_resize1d(path_length, effective_path_num);
// 		
//     // retrieve path
//     for(k=0; k<effective_path_num; k++){
//         path_size = 0;
//         i = THCudaTensor_get1d(end_array, k);//len2-1;
//         j = len1-1;
//         pos = THCudaTensor_get1d(start_array, k);
// 
//         THCudaTensor_set3d(path, k, 0, 0, j);
//         THCudaTensor_set3d(path, k, 0, 1, i);
//         path_size ++;
// 
//         while(i>0 || j>0){
//             if(i==0)
//                 j = j - 1;
//             else if(j==0)
//                 break; //i = i - 1;
// 		    		else{
//                 // confirm starting pos matches then get left or right
//                 if(THCudaTensor_get2d(start_pos, i-1, j-1) == pos){
//                     diag = THCudaTensor_get2d(accumulated_cost, i-1, j-1);
//                 }
// 								else{
//                     diag = INF;
// 								}
// 								if(THCudaTensor_get2d(start_pos, i-1, j) == pos){
//                     left = THCudaTensor_get2d(accumulated_cost, i-1, j);
// 								}
// 								else{
//                     left = INF;
// 								}
//                 if(THCudaTensor_get2d(start_pos, i, j-1) == pos){
//                     down = THCudaTensor_get2d(accumulated_cost, i, j-1);
// 								}
// 								else{
//                     down = INF;
// 								}
//                 tmp = fmin(fmin(diag, left), down);
// 
//                 // obtain path
//                 if(tmp == left){ i = i-1; }
//                 else if(tmp == down){ j = j-1; }
//                 else { i = i-1; j = j-1; }
//  
// 		    		}
//             THCudaTensor_set3d(path, k, path_size, 0, j);
//             THCudaTensor_set3d(path, k, path_size, 1, i);
//             path_size ++;
//         }
//         
//         THCudaTensor_set1d(path_length, k, path_size);
//     }
// 
//     THCudaTensor_free(accumulated_cost);
//     THCudaTensor_free(distances);
//     THCudaTensor_free(start_pos);
//     THCudaTensor_free(start_array);
//     THCudaTensor_free(end_array)  ;
// 
//     return effective_path_num;
// }


