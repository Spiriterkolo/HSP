//
// Created by theogill92 on 11/14/22.
//

#include <stdio.h>

__global__ void cuda_hello() {
    printf("Hello World !\n");
}

void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i][j] = (rand()/RAND_MAX)*2-1;
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    for(int x = 0 ; x < n ; x++) {
        printf(" (");
        for(int y = 0 ; y < p ; y++){
            printf("%d     ", M[x][y]);
        }
        printf(")\n");
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i][j] = M1[i][j] + M2[i][j];
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

    
}

__global__ void matMul(float* d_M, float* d_N, float* d_P, int width, int length) {
    int row = blockIdx.y*width+threadIdx.y;
    int col = blockIdx.x*width+threadIdx.x;
    if(row<width && col <length) {
        float product_val = 0
        for(int k=0;k<width;k++) {
            product_val += d_M[row*width+k]*d_N[k*width+col];
        }
        d_P[row*width+col] = product_val;
    }
}
int main() {
    cuda_hello<<<1,1>>>();

    float *mat1, *mat2, *prod;
    float *d_mat1, *d_mat2, *d_prod;

    // Allocate memory
    mat1   = (float*)malloc(sizeof(float) * M);
    mat2   = (float*)malloc(sizeof(float) * N);
    d_prod = (float*)malloc(sizeof(float) * (M*N));

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaMalloc((void**)&d_a, sizeof(float)*N);
    cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc((void**)&d_out, sizeof(float)*N);

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    matMul(float* [1,2,3], float* [2,4,6], float* d_P, int width);
    cudaDeviceSynchronize();
    return 0 ;
}
