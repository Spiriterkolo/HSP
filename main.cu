#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

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

//Initialisation avec des valeurs qui augmentent avec les lignes et les colonnes
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
        Mout[i] += Min[threadIdx.x + threadIdx.y * blockDim.y + blockIdx.x * blockDim.x * blockDim.y] *
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
        Mout[i] = Min[threadIdx.x] * W[threadIdx.x + i * blockDim.x];
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

    float *raw_data_before_zeropad, *raw_data, *C1_data, *C2_data, *S1_data, *C1_kernel, *S2_data, *C2_kernel, *S5_data, *W1, *B1, *W2, *B2, *W3, *B3;
    float *d_raw_data_before_zeropad, *d_raw_data, *d_C1_data,*d_C2_data, *d_C1_kernel, *d_S1_data, *d_S2_data, *d_C2_kernel, *d_S3_data, *d_S4_data, *d_S5_data, *d_W1, *d_B1, *d_W2, *d_B2, *d_W3, *d_B3;
    dim3 blocks(6, 1, 1);
    dim3 threads(28,28,1);
    dim3 threads2(14,14,1);
    dim3 blocks3(16, 1, 1);
    dim3 threads3(10,10,1);
    dim3 threads4(5,5,1);

    raw_data_before_zeropad = (float*)malloc((sizeof (float))* 28 * 28);
    raw_data = (float*)malloc((sizeof (float))* 32 * 32);
    C1_data = (float*)malloc((sizeof (float))* 6 * 28 *28);
    S1_data = (float*)malloc((sizeof (float))* 6 * 14 * 14);
    C1_kernel = (float*)malloc((sizeof (float))* 6 * 5 * 5);
    C2_data = (float*)malloc((sizeof (float))* 16 * 10 * 10);
    S2_data = (float*)malloc((sizeof (float))* 16 * 5 * 5);
    C2_kernel = (float*)malloc((sizeof (float))* 6 * 5 * 5);
    S5_data = (float*)malloc((sizeof (float))* 10);
    W1 = (float*)malloc((sizeof (float))* 400 * 120);
    W2 = (float*)malloc((sizeof (float))* 120 * 84);
    W3 = (float*)malloc((sizeof (float))* 84 * 10);
    B1 = (float*)malloc((sizeof (float))*120);
    B2 = (float*)malloc((sizeof (float))*84);
    B3 = (float*)malloc((sizeof (float))*10);

    MatrixInit(raw_data_before_zeropad, 28, 28);

    //Pour les mesures de temps d'execution :
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    char buff[100];
    strftime(buff, sizeof buff, "%D %T", gmtime(&ts.tv_sec));
    printf("Current time: %s.%09ld UTC\n", buff, ts.tv_nsec);
    printf("Raw timespec.time_t: %jd\n", (intmax_t)ts.tv_sec);
    printf("Raw timespec.tv_nsec: %09ld\n", ts.tv_nsec);

    MatrixInit0(raw_data, 32, 32);

    struct timespec ts2;
    timespec_get(&ts2, TIME_UTC);
    char buff2[100];
    strftime(buff2, sizeof buff, "%D %T", gmtime(&ts2.tv_sec));
    printf("Current time: %s.%09ld UTC\n", buff2, ts2.tv_nsec);
    printf("Raw timespec.time_t: %jd\n", (intmax_t)ts2.tv_sec);
    printf("Raw timespec.tv_nsec: %09ld\n", ts2.tv_nsec);

    MatrixInit0(C1_data, 6, 28*28);
    MatrixInit0(S1_data, 6, 14*14);
    MatrixInit(C1_kernel, 6, 5*5);
    MatrixInit(C2_kernel, 6, 5*5);

    MatrixInitTest2(W1, 400, 120);
    MatrixInitTest2(B1, 120, 1);
    MatrixInitTest2(W2, 120, 84);
    MatrixInitTest2(B2, 84, 1);
    MatrixInitTest2(W3, 84, 10);
    MatrixInitTest2(B3, 10, 1);

    MatrixPrint(raw_data, 32, 32);
    printf("\n\n");
    MatrixPrint(C1_kernel, 6, 5*5);
    printf("\n\n");

    cudaMalloc((void**)&d_raw_data_before_zeropad, sizeof(float)*(28*28));
    cudaMalloc((void**)&d_raw_data, sizeof(float)*(32*32));
    cudaMalloc((void**)&d_C1_data, sizeof(float)*(6*28*28));
    cudaMalloc((void**)&d_C2_data, sizeof(float)*(16*10*10));
    cudaMalloc((void**)&d_C1_kernel, sizeof(float)*(6*5*5));
    cudaMalloc((void**)&d_S1_data, sizeof(float)*(6*14*14));
    cudaMalloc((void**)&d_S2_data, sizeof(float)*(16*5*5));
    cudaMalloc((void**)&d_C2_kernel, sizeof(float)*(6*5*5));
    cudaMalloc((void**)&d_S3_data, sizeof(float)*(120));
    cudaMalloc((void**)&d_S4_data, sizeof(float)*(84));
    cudaMalloc((void**)&d_S5_data, sizeof(float)*(10));
    cudaMalloc((void**)&d_W1, sizeof(float)*(400*120));
    cudaMalloc((void**)&d_B1, sizeof(float)*(120));
    cudaMalloc((void**)&d_W2, sizeof(float)*(120*84));
    cudaMalloc((void**)&d_B2, sizeof(float)*(84));
    cudaMalloc((void**)&d_W3, sizeof(float)*(84*10));
    cudaMalloc((void**)&d_B3, sizeof(float)*(10));

    cudaMemcpy(d_raw_data_before_zeropad, raw_data_before_zeropad, sizeof(float) * (28*28), cudaMemcpyHostToDevice);
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * (32*32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * (6*28*28), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_data, C2_data, sizeof(float) * (16*10*10), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * (6*5*5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * (6*14*14), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, sizeof(float) * (16*5*5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_kernel, C2_kernel, sizeof(float) * (6*5*5), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S5_data, S5_data, sizeof(float) * (10), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, sizeof(float) * (400*120), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, sizeof(float) * (120*84), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, W3, sizeof(float) * (84*10), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, B1, sizeof(float) * (120), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, B2, sizeof(float) * (84), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B3, B3, sizeof(float) * (10), cudaMemcpyHostToDevice);

    //Net-5's architecture :
    zero_pad<<<28,28>>>(d_raw_data_before_zeropad, d_raw_data);
    cudaConvolution<<<blocks,threads>>>(d_raw_data, d_C1_kernel, d_C1_data);
    cudaDownSampling<<<blocks,threads2>>>(d_C1_data,d_S1_data);
    cudaConvolution<<<blocks3,threads3>>>(d_S1_data, d_C2_kernel, d_C2_data);
    cudaDownSampling<<<blocks3,threads4>>>(d_C2_data,d_S2_data);
    cudaDensetanh1<<<blocks3,threads4>>>(d_S2_data, d_W1, d_B1, d_S3_data);
    cudaDensetanh2<<<1,120>>>(d_S3_data, d_W2, d_B2, d_S4_data);
    cudaDensesoftmax<<<1,84>>>(d_S4_data, d_W3, d_B3, d_S5_data);

    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * (6*28*28), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * (6*14*14), cudaMemcpyDeviceToHost);
    cudaMemcpy(C2_data, d_C2_data, sizeof(float) * (16*10*10), cudaMemcpyDeviceToHost);
    cudaMemcpy(S2_data, d_S2_data, sizeof(float) * (16*5*5), cudaMemcpyDeviceToHost);
    cudaMemcpy(S5_data, d_S5_data, sizeof(float) * (10), cudaMemcpyDeviceToHost);


    //Tested for kernel only composed of 1 and data only composed of 1. Give indeed 25 for each pixel value.
    //Tested for data (0, 1, 2, ... , 31) and it works
    //The dimensions are correct

    MatrixPrint(S5_data, 10, 1);

    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_S2_data);
    cudaFree(d_C2_kernel);
    cudaFree(d_C2_data);
    cudaFree(d_S3_data);
    cudaFree(d_S4_data);
    cudaFree(d_S5_data);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_W3);
    cudaFree(d_B1);
    cudaFree(d_B2);
    cudaFree(d_B3);

    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(S2_data);
    free(C2_kernel);
    free(C2_data);
    free(W1);
    free(W2);
    free(W3);
    free(B1);
    free(B2);
    free(B3);

    return 0;

}