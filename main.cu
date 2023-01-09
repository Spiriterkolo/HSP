#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Initialisation de la matrice avec des valeurs aléatoires entre 0 et 1
void MatrixInit(float *M, int n, int p){
     float max = RAND_MAX;
     for(int i=0; i<n; i++){
         for(int j=0; j<p; j++){
             M[j + i*p] = rand() / max ;
         }
     }
}

//Initialisation avec uniquement des 0
void MatrixInit0(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[j + i * p] = 0 ;
        }
    }
}

//Initialisation avec uniquement des 1
void MatrixInitTest(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[j + i * p] = 1 ;
        }
    }
}

//Initialisation avec des valeurs qui augmentent avec le rang
void MatrixInitTest2(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[j + i * p] = j ;
        }
    }
}

//Permet de print la matrice
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

//Addition de matrices sur CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[j + i * p] = M1[j + i * p] + M2[j + i * p];
        }
    }
}

//Addition de matrices sur GPU
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout){
            Mout[threadIdx.x + blockIdx.x * blockDim.x] = M1[threadIdx.x + blockIdx.x * blockDim.x] + M2[threadIdx.x + blockIdx.x * blockDim.x];
}

//Multiplication de 2 matrices sur CPU
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

//Multiplication de 2 matrices sur GPU
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

//Zero_padding pour la première convolution
__global__ void zero_pad(float *MO, float *Pout){
    Pout[threadIdx.x + 2 + (blockIdx.x+2) * (blockDim.x+4)] = MO[threadIdx.x + blockIdx.x * blockDim.x];
}

//Layer de convolution
__global__ void cudaConvolution(float *data, float *kernel, float *Cout){
    float sum = 0;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int k = blockIdx.x;
    for (int j = 0; j < 5; j++){
        for (int i = 0; i < 5; i++){
            sum += data[(x + i)+ (y + j) * (blockDim.x + 4)] * kernel[i + j * 5 + k * 5 * 5];
        }
    }
    Cout[x + y * blockDim.x + k * blockDim.x * blockDim.y] = sum;
}

//Fonction d'activation tanh
__device__ float activation_tanh(float M) {
    return tanhf(M);
}

//Fonction d'activation softmax
__device__ void activation_softmax(float *M, float max){
    int shape = int(sizeof(*M)/ sizeof(max));
    float sum = 0;
    float esum;
    for(int i=0; i < shape; i++){
        sum += M[i];
    }
    esum = expf(sum-max);
    M[threadIdx.x] = expf(M[threadIdx.x]-max) / esum;
}

//Pooling layer
__global__ void cudaDownSampling(float *Conved, float *Cout){
    float mean = 0;
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            mean += Conved[(2 * threadIdx.x + i) + (2 * threadIdx.y + j) * 2 * blockDim.x +  blockIdx.x * 4 * blockDim.x * blockDim.y];
        }
    }
    Cout[threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.x] = activation_tanh(mean/4);
}

//Premier dense layer
__global__ void cudaDensetanh1(float *Min, float *W, float *B, float *Mout) {
    for (int i = 0; i < 120; i++) {
        Mout[i] = Min[threadIdx.x + threadIdx.y * blockDim.y + blockIdx.x * blockDim.x * blockDim.y] *
                  W[threadIdx.x + threadIdx.y * blockDim.y + blockIdx.x * blockDim.x * blockDim.y +
                    i * gridDim.x * blockDim.x * blockDim.y];
    }
    for (int j = 0; j < 120; j++){
        Mout[j] = activation_tanh(Mout[j]+B[j]);
    }
}

//Second dense layer
__global__ void cudaDensetanh2(float *Min, float *W, float *B, float *Mout) {
    for (int i = 0; i < 84; i++) {
        Mout[i] = Min[threadIdx.x] *
                  W[threadIdx.x + i * blockDim.x];
    }
    for (int j = 0; j < 84; j++){
        Mout[j] = activation_tanh(Mout[j]+B[j]);
    }
}

//Troisième Dense layer
__global__ void cudaDensesoftmax(float *Min, float *W, float *B, float *Mout) {
    for (int i = 0; i < 10; i++) {
        Mout[i] = Min[threadIdx.x] *
                  W[threadIdx.x + i * blockDim.x];
    }
    float max = Mout[0];
    for (int j = 0; j < 10; j++) {
        Mout[j] += B[j];
        if(Mout[j] > max){
            max = Mout[j];
        }
    }
    activation_softmax(Mout, max);
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


    MatrixInit(raw_data, 32, 32);
    MatrixInit0(C1_data, 6, 28*28);
    MatrixInit0(S1_data, 6, 14*14);
    MatrixInit(C1_kernel, 6, 5*5);
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
    //Tested for kernel only composed of 1 and data only composed of 1. Give indeed 25 for each pixel value.
    //Tested for data (0, 1, 2, ... , 31) and it works
    //The dimensions are correct
    MatrixPrint(S1_data, 6, 14*14);
    //Have the correct dimensions. Checked that the value of a pixel is the mean of the 4 corresponding pixels

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