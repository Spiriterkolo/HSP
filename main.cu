//
// Created by theogill92 on 11/14/22.
//

//
// Created by theogill92 on 11/14/22.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void MatrixInit1(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[j + i * p] = 1 ;
        }
    }
}

void MatrixInitTest(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[j + i * p] = j ;
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

__device__ float activation_tanh(float M){
    return tanhf(M);
}

__device__ float activation_softmax(float *M){

}

__global__ void cudaConvolution(float *data, float *kernel, float *Cout){
    float sum = 0;
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
            sum += data[(threadIdx.x + i) + (threadIdx.y + j) * (blockDim.x+4)] * kernel[i + j * 5 + blockIdx.x * 5 * 5];
        }
    }
    Cout[threadIdx.x + threadIdx.y * blockDim.y + blockIdx.x * blockDim.x * blockDim.y] = activation_tanh(sum);
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

// Pas besoin de Flatten mais il faut s'assurer que les poids dans Python correspondent bien aux bonnes entrées pour le premier Dense.

__global__ void cudaDensetanh(float *Mentree, float *W, float *B float *Msortie) {
    //cudaMatrixMult(W, Mentree, Msortie);
    //cudaMatrixAdd(Msortie, B, Msortie);
    float sum = 0;
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
            sum += data[(threadIdx.x + i) + (threadIdx.y + j) * (blockDim.x+4)] * kernel[i + j * 5 + blockIdx.x * 5 * 5];
        }
    }
    Msortie[threadIdx.x + threadIdx.y * blockDim.y + blockIdx.x * blockDim.x * blockDim.y] = activation_tanh(sum);
}

__global__ void cudaDensesoftmax(float *Mentree, float *W, float *B float *Msortie) {
    //cudaMatrixMult(Mentree, W, Msortie)
    //cudaMatrixAdd(Msortie, B, Msortie)
    float sum = 0;
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
            sum += W[i + j * 5 + blockIdx.x * 5 * 5] * Mentree[(threadIdx.x + i) + (threadIdx.y + j) * (blockDim.x+4)];
        }
    }
    Msortie[threadIdx.x + threadIdx.y * blockDim.y + blockIdx.x * blockDim.x * blockDim.y] = activation_softmax(sum);
}
int main(){
    //float *M;
    //M = (float*)malloc((sizeof (float))* n * p);
    //MatrixInit(M, n, p);
    //MatrixPrint(M, n, p);
    //free(M);

    float *raw_data, *C1_data, *C2_data, *S1_data, *C1_kernel, *S2_data, *C2_kernel;
    float *d_raw_data, *d_C1_data,*d_C2_data, *d_C1_kernel, *d_S1_data, *d_S2_data, *d_C2_kernel;
    dim3 blocks(6, 1, 1);
    dim3 threads(28,28,1);
    dim3 threads2(14,14,1);
    dim3 blocks3(16, 1, 1);
    dim3 threads3(10,10,1);
    dim3 threads4(5,5,1);

    raw_data = (float*)malloc((sizeof (float))* 32 * 32);
    C1_data = (float*)malloc((sizeof (float))* 6 * 28 *28);
    S1_data = (float*)malloc((sizeof (float))* 6 * 14 * 14);
    C1_kernel = (float*)malloc((sizeof (float))* 6 * 5 * 5);
    C2_data = (float*)malloc((sizeof (float))* 16 * 10 * 10);
    S2_data = (float*)malloc((sizeof (float))* 16 * 5 * 5);
    C2_kernel = (float*)malloc((sizeof (float))* 6 * 5 * 5);


    MatrixInit(raw_data, 32, 32);
    MatrixInit0(C1_data, 6, 28*28);
    MatrixInit0(C2_data, 16, 5*5);
    MatrixInit0(S1_data, 6, 14*14);
    MatrixInit0(S2_data, 16, 5*5);
    MatrixInit(C1_kernel, 6, 5*5);
    MatrixInit(C2_kernel, 6, 5*5);

    //MatrixPrint(raw_data, 32, 32);
    //printf("\n\n");
    //MatrixPrint(C1_kernel, 6, 5*5);
    //printf("\n\n");

    cudaMalloc((void**)&d_raw_data, sizeof(float)*(32*32));
    cudaMalloc((void**)&d_C1_data, sizeof(float)*(6*28*28));
    cudaMalloc((void**)&d_C2_data, sizeof(float)*(16*10*10));
    cudaMalloc((void**)&d_C1_kernel, sizeof(float)*(6*5*5));
    cudaMalloc((void**)&d_S1_data, sizeof(float)*(6*14*14));
    cudaMalloc((void**)&d_S2_data, sizeof(float)*(16*5*5));
    cudaMalloc((void**)&d_C2_kernel, sizeof(float)*(6*5*5));

    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * (32*32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * (6*28*28), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_data, C2_data, sizeof(float) * (16*10*10), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * (6*5*5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * (6*14*14), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, sizeof(float) * (16*5*5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_kernel, C2_kernel, sizeof(float) * (6*5*5), cudaMemcpyHostToDevice);

    cudaConvolution<<<blocks,threads>>>(d_raw_data, d_C1_kernel, d_C1_data);
    cudaDownSampling<<<blocks,threads2>>>(d_C1_data,d_S1_data);
    cudaConvolution<<<blocks3,threads3>>>(d_S1_data, d_C2_kernel, d_C2_data);
    cudaDownSampling<<<blocks3,threads4>>>(d_C2_data,d_S2_data);

    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * (6*28*28), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * (6*14*14), cudaMemcpyDeviceToHost);
    cudaMemcpy(C2_data, d_C2_data, sizeof(float) * (16*10*10), cudaMemcpyDeviceToHost);
    cudaMemcpy(S2_data, d_S2_data, sizeof(float) * (16*5*5), cudaMemcpyDeviceToHost);

    //MatrixPrint(C1_data, 6, 28*28);
    //MatrixPrint(S1_data, 6, 14*14);

    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_S2_data);
    cudaFree(d_C2_kernel);
    cudaFree(d_C2_data);

    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(S2_data);
    free(C2_kernel);
    free(C2_data);
    return 0;
}

