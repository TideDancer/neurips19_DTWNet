// to support python 3 where all ints are long
#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong PyLong_AsLong
#endif

#include <TH/TH.h>
#include <math.h>

#define INF 999999

int dynamic_prog_lib_forward(const THFloatTensor *input1, const THFloatTensor *input2, THIntTensor* path)
// input1: kernel, must be tensor of size (k,) where k is the length
// input2: subseq, must be tensor of size (s,) where s is the length
{
    int i, j, path_size;
    double tmp;
    int len2 = THFloatTensor_size(input2, 0);
    int len1 = THFloatTensor_size(input1, 0);

    // ---------------------- compute dtw -------------------

    THFloatTensor* accumulated_cost = THFloatTensor_newWithSize2d(len2, len1);
    THFloatTensor* distances = THFloatTensor_newWithSize2d(len2, len1);

    for(i=0; i<len2; i++)
        for(j=0; j<len1; j++){
            tmp = (THFloatTensor_get1d(input1, j) - THFloatTensor_get1d(input2, i))*(THFloatTensor_get1d(input1, j) - THFloatTensor_get1d(input2, i));
            THFloatTensor_set2d(distances, i, j, tmp);
    }
    THFloatTensor_set2d(accumulated_cost, 0, 0, THFloatTensor_get2d(distances, 0, 0));
 
		for(i=1; i<len1; i++){
        tmp = THFloatTensor_get2d(distances, 0, i) + THFloatTensor_get2d(accumulated_cost, 0, i-1);
        THFloatTensor_set2d(accumulated_cost, 0, i, tmp);  
    }
		for(i=1; i<len2; i++){
        tmp = THFloatTensor_get2d(distances, i, 0) + THFloatTensor_get2d(accumulated_cost, i-1, 0);
        THFloatTensor_set2d(accumulated_cost, i, 0, tmp);
		}
  
    for(i=1; i<len2; i++)
				for(j=1; j<len1; j++){
            tmp = fmin(THFloatTensor_get2d(accumulated_cost, i-1, j-1), THFloatTensor_get2d(accumulated_cost, i-1, j));
            tmp = fmin(tmp, THFloatTensor_get2d(accumulated_cost, i, j-1));
            tmp = tmp + THFloatTensor_get2d(distances, i, j);
            THFloatTensor_set2d(accumulated_cost, i, j, tmp);
		}

    // ---------------------- obtain path -------------------
    path_size = 0;
    i = len2-1;
    j = len1-1;

    THIntTensor_set2d(path, 0, 0, j);
    THIntTensor_set2d(path, 0, 1, i);
    path_size ++;

    while(i>0 || j>0){
        if(i==0)
            j = j - 1;
        else if(j==0)
            i = i - 1;
				else{
            tmp = fmin(THIntTensor_get2d(accumulated_cost, i-1, j-1), THIntTensor_get2d(accumulated_cost, i-1, j));
            tmp = fmin(tmp, THIntTensor_get2d(accumulated_cost, i, j-1));
            if(tmp == THIntTensor_get2d(accumulated_cost, i-1, j)){
                i = i - 1;
            }
						else if(tmp == THIntTensor_get2d(accumulated_cost, i, j-1)){ 
                j = j - 1; 
            }
						else{
                i = i - 1;
                j = j - 1;
						}
				}
        THIntTensor_set2d(path, path_size, 0, j);
        THIntTensor_set2d(path, path_size, 1, i);
        path_size ++;
    }

    THFloatTensor_free(accumulated_cost);
    THFloatTensor_free(distances);
    return path_size;
}


THFloatTensor* dynamic_prog_lib_forward_spring(THFloatTensor *input1, THFloatTensor *input2, int path_num, THIntTensor *path, THIntTensor *path_length)
//THFloatTensor* dynamic_prog_lib_forward_spring(THFloatTensor *input1, THFloatTensor *input2, int path_num, THIntTensor *path, THIntTensor *path_length, THIntTensor *start_pos, THFloatTensor *accumulated_cost) // for debug purpose
// input1: kernel, must be tensor of size (k,) where k is the length
// input2: subseq, must be tensor of size (s,) where s is the length
{
    int i, j, k, pos, end, diag_pos, left_pos, down_pos, path_size;
    double tmp, dtw_result, diag, left, down;
    int len2 = THFloatTensor_size(input2, 0);
    int len1 = THFloatTensor_size(input1, 0);

    // ---------------------- compute dtw -------------------

    THFloatTensor* accumulated_cost = THFloatTensor_newWithSize2d(len2, len1);
    THFloatTensor* distances = THFloatTensor_newWithSize2d(len2, len1);
    THIntTensor* start_pos = THIntTensor_newWithSize2d(len2, len1);

    for(i=0; i<len2; i++)
        for(j=0; j<len1; j++){
            tmp = (THFloatTensor_get1d(input1, j) - THFloatTensor_get1d(input2, i))*(THFloatTensor_get1d(input1, j) - THFloatTensor_get1d(input2, i));
            THFloatTensor_set2d(distances, i, j, tmp);
    }
    THFloatTensor_set2d(accumulated_cost, 0, 0, THFloatTensor_get2d(distances, 0, 0));
    THIntTensor_set2d(start_pos, 0, 0, 0);
 
		for(i=1; i<len1; i++){
        tmp = THFloatTensor_get2d(distances, 0, i) + THFloatTensor_get2d(accumulated_cost, 0, i-1);
        THFloatTensor_set2d(accumulated_cost, 0, i, tmp);  
        THIntTensor_set2d(start_pos, 0, i, 0);
    }
		for(i=1; i<len2; i++){
        tmp = THFloatTensor_get2d(distances, i, 0);
        THFloatTensor_set2d(accumulated_cost, i, 0, tmp);
        THIntTensor_set2d(start_pos, i, 0, i);
		}
  
    for(i=1; i<len2; i++)
				for(j=1; j<len1; j++){
            // compute distance matrix
            diag = THFloatTensor_get2d(accumulated_cost, i-1, j-1);
            left = THFloatTensor_get2d(accumulated_cost, i-1, j);
            down = THFloatTensor_get2d(accumulated_cost, i, j-1);
            tmp = fmin(fmin(diag, left), down);
            THFloatTensor_set2d(accumulated_cost, i, j, tmp+THFloatTensor_get2d(distances, i, j));

            // compute augmented starting_position matrix
            diag_pos = THIntTensor_get2d(start_pos, i-1, j-1);
            left_pos = THIntTensor_get2d(start_pos, i-1, j);
            down_pos = THIntTensor_get2d(start_pos, i, j-1);
            if(tmp==diag) pos = diag_pos;
            else if(tmp==left) pos = left_pos;
            else pos = down_pos;
            THIntTensor_set2d(start_pos, i, j, pos);
		}
		
    // ---------------------- obtain path -------------------
    THFloatTensor* modify_array = THFloatTensor_newWithSize1d(len2);
    THFloatTensor* dtw_results = THFloatTensor_newWithSize1d(path_num);
    THIntTensor* start_array = THFloatTensor_newWithSize1d(path_num);
    THIntTensor* end_array   = THFloatTensor_newWithSize1d(path_num);

		for(i=0; i<len2; i++){
        THFloatTensor_set1d(modify_array, i, THFloatTensor_get2d(accumulated_cost, i, len1-1));
		}
		for(k=0; k<path_num; k++){
        float minval = INF;
        for(i=0; i<len2; i++){
            tmp = THFloatTensor_get1d(modify_array, i);
						if(tmp < minval){
                minval = tmp;
                pos = THIntTensor_get2d(start_pos, i, len1-1);
                end = i;
						}
		    }
				for(i=0; i<len2; i++){
						if(pos == THIntTensor_get2d(start_pos, i, len1-1)){
                THFloatTensor_set1d(modify_array, i, INF);
						}
				}
        THFloatTensor_set1d(dtw_results, k, minval);
        THIntTensor_set1d(start_array, k, pos);
        THIntTensor_set1d(end_array,   k, end);
		}
		
    for(k=0; k<path_num; k++){

        path_size = 0;
        i = THIntTensor_get1d(end_array, k);//len2-1;
        j = len1-1;
        pos = THIntTensor_get1d(start_array, k);

        THIntTensor_set3d(path, k, 0, 0, j);
        THIntTensor_set3d(path, k, 0, 1, i);
        path_size ++;

        while(i>0 || j>0){
            if(i==0)
                j = j - 1;
            else if(j==0)
                break; //i = i - 1;
		    		else{
                // confirm starting pos matches then get left or right
                if(THIntTensor_get2d(start_pos, i-1, j-1) == pos){
                    diag = THFloatTensor_get2d(accumulated_cost, i-1, j-1);
                }
								else{
                    diag = INF;
								}
								if(THIntTensor_get2d(start_pos, i-1, j) == pos){
                    left = THFloatTensor_get2d(accumulated_cost, i-1, j);
								}
								else{
                    left = INF;
								}
                if(THIntTensor_get2d(start_pos, i, j-1) == pos){
                    down = THFloatTensor_get2d(accumulated_cost, i, j-1);
								}
								else{
                    down = INF;
								}
                tmp = fmin(fmin(diag, left), down);

                // obtain path
                if(tmp == left){ i = i-1; }
                else if(tmp == down){ j = j-1; }
                else { i = i-1; j = j-1; }
 
		    		}
            THIntTensor_set3d(path, k, path_size, 0, j);
            THIntTensor_set3d(path, k, path_size, 1, i);
            path_size ++;
        }
        
        THIntTensor_set1d(path_length, k, path_size);

        //for(i=0; i<path_size; i++){printf("%d, %d, %d\n", i, THIntTensor_get2d(path, i, 0), THIntTensor_get2d(path, i, 1));}
    }

    THFloatTensor_free(accumulated_cost);
    THFloatTensor_free(distances);
    THIntTensor_free(start_pos);
    THFloatTensor_free(modify_array);
    THFloatTensor_free(dtw_results);
    THIntTensor_free(start_array);
    THIntTensor_free(end_array)  ;

    return dtw_results;
}



int dynamic_prog_lib_forward_spring_epspath(THFloatTensor *input1, THFloatTensor *input2, float eps, THIntTensor *path, THIntTensor *path_length)
//int dynamic_prog_lib_forward_spring_epspath(THFloatTensor *input1, THFloatTensor *input2, float eps, THIntTensor *path, THIntTensor *path_length, THFloatTensor* accumulated_cost, THIntTensor* start_pos)
// input1: kernel, must be tensor of size (k,) where k is the length
// input2: subseq, must be tensor of size (s,) where s is the length
{
    int i, j, k, pos, end, diag_pos, left_pos, down_pos, path_size;
    double tmp, dtw_result, diag, left, down;
    int len2 = THFloatTensor_size(input2, 0);
    int len1 = THFloatTensor_size(input1, 0);

    // ---------------------- compute dtw -------------------

    THFloatTensor* distances = THFloatTensor_newWithSize2d(len2, len1);
    THFloatTensor* accumulated_cost = THFloatTensor_newWithSize2d(len2, len1);
    THIntTensor* start_pos = THIntTensor_newWithSize2d(len2, len1);

    for(i=0; i<len2; i++)
        for(j=0; j<len1; j++){
            tmp = (THFloatTensor_get1d(input1, j) - THFloatTensor_get1d(input2, i))*(THFloatTensor_get1d(input1, j) - THFloatTensor_get1d(input2, i));
            THFloatTensor_set2d(distances, i, j, tmp);
    }
    THFloatTensor_set2d(accumulated_cost, 0, 0, THFloatTensor_get2d(distances, 0, 0));
    THIntTensor_set2d(start_pos, 0, 0, 0);
 
		for(i=1; i<len1; i++){
        tmp = THFloatTensor_get2d(distances, 0, i) + THFloatTensor_get2d(accumulated_cost, 0, i-1);
        THFloatTensor_set2d(accumulated_cost, 0, i, tmp);  
        THIntTensor_set2d(start_pos, 0, i, 0);
    }
		for(i=1; i<len2; i++){
        tmp = THFloatTensor_get2d(distances, i, 0);
        THFloatTensor_set2d(accumulated_cost, i, 0, tmp);
        THIntTensor_set2d(start_pos, i, 0, i);
		}
  
    for(i=1; i<len2; i++)
				for(j=1; j<len1; j++){
            // compute distance matrix
            diag = THFloatTensor_get2d(accumulated_cost, i-1, j-1);
            left = THFloatTensor_get2d(accumulated_cost, i-1, j);
            down = THFloatTensor_get2d(accumulated_cost, i, j-1);
            tmp = fmin(fmin(diag, left), down);
            THFloatTensor_set2d(accumulated_cost, i, j, tmp+THFloatTensor_get2d(distances, i, j));

            // compute augmented starting_position matrix
            diag_pos = THIntTensor_get2d(start_pos, i-1, j-1);
            left_pos = THIntTensor_get2d(start_pos, i-1, j);
            down_pos = THIntTensor_get2d(start_pos, i, j-1);
            if(tmp==diag) pos = diag_pos;
            else if(tmp==left) pos = left_pos;
            else pos = down_pos;
            THIntTensor_set2d(start_pos, i, j, pos);
		}
		
    // ---------------------- obtain path -------------------
    float thres; int path_num = 1; int effective_path_num = 0;
    float tmp1; int pos1; int flag=0;

    // search for the minimum value and number of path
    thres = THFloatTensor_get2d(accumulated_cost, 0, len1-1);
    pos = THIntTensor_get2d(start_pos, 0, len1-1);
		for(i=1; i<len2; i++){
        tmp = THFloatTensor_get2d(accumulated_cost, i, len1-1);
        pos1 = THIntTensor_get2d(start_pos, i, len1-1);
				if(tmp<thres){ // obtain minimum value
            thres = tmp;
				}
				if(pos1!=pos){ // count number of path
            path_num ++;
            pos = pos1;
				}
		}
    thres = thres * (1+eps);

    THIntTensor* start_array = THIntTensor_newWithSize1d(path_num);
    THIntTensor* end_array   = THIntTensor_newWithSize1d(path_num);

    // obtain each path
    flag = 0;
    tmp = THFloatTensor_get2d(accumulated_cost, 0, len1-1);
    pos = THIntTensor_get2d(start_pos, 0, len1-1);
    end = 0; 
    effective_path_num = 0; i = 1;
		while( i<len2 ){
        if(tmp < thres) flag = 1;
        tmp1 = THFloatTensor_get2d(accumulated_cost, i, len1-1);
        pos1 = THIntTensor_get2d(start_pos, i, len1-1);
        
				if(tmp1 < thres && tmp1 < tmp && pos1 == pos){
            tmp = tmp1;
            end = i;
            flag = 1;
				}
				else if(pos1 != pos && flag == 1){
            // save
            flag = 0;
            //THFloatTensor_set1d(dtw_results, effective_path_num, tmp);
            THIntTensor_set1d(start_array, effective_path_num, pos);
            THIntTensor_set1d(end_array,   effective_path_num, end);
            effective_path_num ++;
            // renew
            tmp = tmp1;
            pos = pos1;
            end = i;
				}
				else if(pos1 != pos && flag == 0){
            // renew
            tmp = tmp1;
            pos = pos1;
            end = i;
            if(tmp < thres) flag = 1;
				}
        i++;
		}
		if(flag == 1){ // handle the possible last effective path 
        THIntTensor_set1d(start_array, effective_path_num, pos);
        THIntTensor_set1d(end_array,   effective_path_num, end);
        effective_path_num ++;
		}

		// if(effective_path_num==0){
    //     printf("%f,%f\n",thres,thres/(1+eps));
    //     for(i=0;i<len2;i++) printf("%f,", THFloatTensor_get2d(accumulated_cost,i,len1-1));
    //     printf("\n");
    //     for(i=0;i<len2;i++) printf("%d,", THIntTensor_get2d(start_pos,i,len1-1));
		// }

    // resize the path and path_length matrix
    THIntTensor_resize3d(path, effective_path_num, len1+len2, 2);
    THIntTensor_resize1d(path_length, effective_path_num);
		
    // retrieve path
    for(k=0; k<effective_path_num; k++){
        path_size = 0;
        i = THIntTensor_get1d(end_array, k);//len2-1;
        j = len1-1;
        pos = THIntTensor_get1d(start_array, k);

        THIntTensor_set3d(path, k, 0, 0, j);
        THIntTensor_set3d(path, k, 0, 1, i);
        path_size ++;

        while(i>0 || j>0){
            if(i==0)
                j = j - 1;
            else if(j==0)
                break; //i = i - 1;
		    		else{
                // confirm starting pos matches then get left or right
                if(THIntTensor_get2d(start_pos, i-1, j-1) == pos){
                    diag = THFloatTensor_get2d(accumulated_cost, i-1, j-1);
                }
								else{
                    diag = INF;
								}
								if(THIntTensor_get2d(start_pos, i-1, j) == pos){
                    left = THFloatTensor_get2d(accumulated_cost, i-1, j);
								}
								else{
                    left = INF;
								}
                if(THIntTensor_get2d(start_pos, i, j-1) == pos){
                    down = THFloatTensor_get2d(accumulated_cost, i, j-1);
								}
								else{
                    down = INF;
								}
                tmp = fmin(fmin(diag, left), down);

                // obtain path
                if(tmp == left){ i = i-1; }
                else if(tmp == down){ j = j-1; }
                else { i = i-1; j = j-1; }
 
		    		}
            THIntTensor_set3d(path, k, path_size, 0, j);
            THIntTensor_set3d(path, k, path_size, 1, i);
            path_size ++;
        }
        
        THIntTensor_set1d(path_length, k, path_size);
    }

    THFloatTensor_free(accumulated_cost);
    THFloatTensor_free(distances);
    THIntTensor_free(start_pos);
    THIntTensor_free(start_array);
    THIntTensor_free(end_array)  ;

    return effective_path_num;
}


