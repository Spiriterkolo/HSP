//
// Created by theogill92 on 11/14/22.
//

#include <stdio.h>

__global__ void cuda_hello() {
    printf("Hello World !\n");
}

__global__ void matMul(float* d_M, float* d_N, float* d_P, int width) {
    int row = blockIdx.y*width+threadIdx.y;
    int col = blockIdx.x*width+threadIdx.x;
    if(row<width && col <width) {
        float product_val = 0
        for(int k=0;k<width;k++) {
            product_val += d_M[row*width+k]*d_N[k*width+col];
        }
        d_P[row*width+col] = product_val;
    }
}
int main() {
    cuda_hello<<<1,1>>>();
    matMul(float* [1,2,3], float* [2], float* d_P, int width);
    cudaDeviceSynchronize();
    return 0 ;
}
