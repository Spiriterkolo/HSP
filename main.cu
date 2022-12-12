#include <stdio.h>
#include <stdlib.h>

void MatrixInit(float *M, int n, int p){
     float max = RAND_MAX;
     for(int i=0; i<n; i++){
         for(int j=0; j<p; j++){
             M[j + i*p] = rand() / max ;
         }
     }
}

void MatrixInit0(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[j + i * p] = 0 ;
        }
    }
}

void MatrixInitTest(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[j + i * p] = 1 ;
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    printf("Matrice : ");
    for(int i=0; i<n; i++){
        printf("\n");
        for(int j=0; j<p; j++){
            printf("%.2f ", M[j + i*p]);
        }
    }
    printf("\n");
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[j + i * p] = M1[j + i * p] + M2[j + i * p];
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout){
    for (int i = 0; i < gridDim.x; i++){
        for (int j = 0; j < blockDim.x; j++){
            Mout[j + i * blockDim.x] = M1[j + i * blockDim.x] + M2[j + i * blockDim.x];
        }
    }
}

__global__ void cudaConvolution(float *data, float *kernel, float *Cout){
    float sum = 0;
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
            sum += data[(threadIdx.y + j) + (threadIdx.x + i) * blockDim.y] * kernel[j + i * 5 + blockIdx.x * 5 * 5];
        }
    }
    Cout[threadIdx.y + threadIdx.x * blockDim.y + blockIdx.x * blockDim.x * blockDim.y] = sum;
}

__global__ void cudaDownSampling(float *Conved, float *Cout){
    float mean = 0;
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            mean += Conved[(2 * threadIdx.y + j) + (2 * threadIdx.x + i) * 2 * blockDim.y +  blockIdx.x * 4 * blockDim.x * blockDim.y];
        }
    }
    Cout[threadIdx.y + threadIdx.x * blockDim.x + blockIdx.x * blockDim.x * blockDim.x] = mean/4;
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    float sum;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum = 0;
            for (int k = 0; k < n; k++) {
                sum += M1[j+i*n+k]*M2[j+(i+k)*n];
            }
            Mout[j+i*n] = sum;
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout){
    float sum;
    for (int i = 0; i < gridDim.x; i++) {
        for (int j = 0; j < gridDim.x; j++) {
            sum = 0;
            for (int k = 0; k < gridDim.x; k++) {
                sum += M1[j+i*gridDim.x+k]*M2[j+(i+k)*gridDim.x];
            }
            Mout[j+i*gridDim.x] = sum;
        }
    }
}

int main(){
    //float *M;
    //M = (float*)malloc((sizeof (float))* n * p);
    //MatrixInit(M, n, p);
    //MatrixPrint(M, n, p);
    //free(M);

    float *raw_data, *C1_data, *S1_data, *C1_kernel;
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data;
    dim3 blocks(6, 1, 1);
    dim3 threads(28,28,1);
    dim3 threads2(14,14,1);

    raw_data = (float*)malloc((sizeof (float))* 32 * 32);
    C1_data = (float*)malloc((sizeof (float))* 6 * 28 *28);
    S1_data = (float*)malloc((sizeof (float))* 6 * 14 * 14);
    C1_kernel = (float*)malloc((sizeof (float))* 6 * 5 * 5);


    MatrixInitTest(raw_data, 32, 32);
    MatrixInit0(C1_data, 6, 28*28);
    MatrixInit0(S1_data, 6, 14*14);
    MatrixInitTest(C1_kernel, 6, 5*5);
    MatrixPrint(raw_data, 32, 32);
    printf("\n\n");
    MatrixPrint(C1_kernel, 6, 5*5);
    printf("\n\n");

    cudaMalloc((void**)&d_raw_data, sizeof(float)*(32*32));
    cudaMalloc((void**)&d_C1_data, sizeof(float)*(6*28*28));
    cudaMalloc((void**)&d_C1_kernel, sizeof(float)*(6*5*5));
    cudaMalloc((void**)&d_S1_data, sizeof(float)*(6*14*14));

    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * (32*32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * (6*28*28), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * (6*5*5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * (6*14*14), cudaMemcpyHostToDevice);

    cudaConvolution<<<blocks,threads>>>(d_raw_data, d_C1_kernel, d_C1_data);
    cudaDownSampling<<<blocks,threads2>>>(d_C1_data,d_S1_data);

    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * (6*28*28), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * (6*14*14), cudaMemcpyDeviceToHost);

    MatrixPrint(C1_data, 6, 28*28);
    //MatrixPrint(S1_data, 6, 14*14);

    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);

    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    return 0;
}