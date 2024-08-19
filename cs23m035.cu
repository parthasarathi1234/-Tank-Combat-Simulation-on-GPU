#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void round_inc(int *gpu_r, int *gpu_n, int T){
    gpu_r[0]++;
    gpu_n[0]=0;
    if(gpu_r[0]%T==0){
        gpu_r[0]++;
    }
}

__global__ void initialization(int H, int *gpu_hp, int *gpu_active, int *gpu_score, int *gpu_r, int T){
    gpu_r[0]=1;

    int id=threadIdx.x;
    gpu_hp[id]=H;
    gpu_active[id]=1;
    gpu_score[id]=0;
}

__global__ void active(int *gpu_active, int *gpu_hp, int *gpu_n){
    if(gpu_hp[threadIdx.x]<=0){
        gpu_active[threadIdx.x]=0;
    }
    else{
        atomicAdd(&gpu_n[0],1);
    }
    
}

__global__ void game(int T, int *gpu_r, int M, int N,int *gpu_xcoord, int *gpu_ycoord, int *gpu_score, int *gpu_hp, int *gpu_active, int *gpu_n){
  
    int t_id=threadIdx.x;
    int b_id=blockIdx.x;

    __shared__ int shared_distance;
    shared_distance=M+N+1;
    
     __syncthreads();

    int temp=0;
    if(gpu_active[b_id]==1 && b_id!=t_id && gpu_active[t_id]==1){

        int target_tank=(b_id+gpu_r[0])%T;

        int x_ft=gpu_xcoord[b_id];
        int y_ft=gpu_ycoord[b_id];

        int x_tt=gpu_xcoord[target_tank];
        int y_tt=gpu_ycoord[target_tank];

        int x_chan=x_tt-x_ft;
        int y_chan=y_tt-y_ft;

        int a=gpu_xcoord[t_id]-x_ft;
        int b=gpu_ycoord[t_id]-y_ft;

        if(((x_chan<0 && a<0) || (y_chan<0 && b<0) || (x_chan>0 && a>0) || (y_chan>0 && b>0)) &&  a*y_chan==b*x_chan)
        {   
            int aa=(a>0)?a:-a;
            int bb=(b>0)?b:-b;
            temp=aa+bb;
            atomicMin(&shared_distance,temp);        
        }
    }

    __syncthreads();
    if(shared_distance==temp){
        gpu_score[b_id]+=1;
        atomicSub(&gpu_hp[t_id],1);
    }    
}
//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    
    int *gpu_xcoord;
    int *gpu_ycoord;
    int *gpu_score;
    int *gpu_hp;
    int *gpu_active;
    int *gpu_r;
    int *gpu_n;

    cudaMalloc(&gpu_xcoord,sizeof(int)*T);
    cudaMemcpy(gpu_xcoord, xcoord,sizeof(int)*T,cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_ycoord, sizeof(int)*T);
    cudaMemcpy(gpu_ycoord, ycoord, sizeof(int)*T, cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_score, sizeof(int)*T);

    cudaMalloc(&gpu_hp, sizeof(int)*T);

    cudaMalloc(&gpu_active, sizeof(int)*T);

    cudaMalloc(&gpu_r, sizeof(int)*1);
    cudaMalloc(&gpu_n, sizeof(int)*1);

    initialization<<<1,T>>>(H, gpu_hp, gpu_active, gpu_score, gpu_r, T);

    int count=T;
    cudaDeviceSynchronize();
    while(count>1){
        game<<<T,T>>>(T, gpu_r, M,N,gpu_xcoord, gpu_ycoord, gpu_score, gpu_hp, gpu_active,gpu_n);
        round_inc<<<1,1>>>(gpu_r, gpu_n, T);
        active<<<1,T>>>(gpu_active, gpu_hp, gpu_n);
        cudaMemcpy(&count,gpu_n,sizeof(int)*1,cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(score, gpu_score, sizeof(int)*T, cudaMemcpyDeviceToHost);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
