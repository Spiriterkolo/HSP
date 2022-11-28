//
// Created by theogill92 on 11/14/22.
//

#include <stdio.h>

__global__ void cuda_hello() {
    printf("Hello World !\n");
}

void MatrixInit(float* M, int n, int p) {
    float max = RAND_MAX;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[j+i*p] = (rand()/max)*2-1;
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    for(int x = 0 ; x < n ; x++) {
        printf(" (");
        for(int y = 0 ; y < p ; y++){
            printf("%f     ", M[y+x*p]);
        }
        printf(")\n");
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[j+i*p] = M1[j+i*p] + M2[j+i*p];
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout) {
    for (int i = 0; i < gridDim.x; i++) {
        for (int j = 0; j < blockDim.x; j++) {
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

int main() {
    cuda_hello<<<1,1>>>();
    int N = 3;
    int P = 3;
    float *Mat1, *Mat2, *MatOut;
    float *d_Mat1, *d_Mat2, *d_MatOut;


    Mat1 = (float*)malloc(sizeof(float) * (N*P));
    Mat2 = (float*)malloc(sizeof(float) * (N*P));
    MatOut = (float*)malloc(sizeof(float) * (N*P));

    MatrixInit(Mat1, N, P);
    MatrixInit(Mat2, N, P);

    MatrixPrint(Mat1, N, P);
    MatrixPrint(Mat2, N, P);

    MatrixAdd(Mat1, Mat2, MatOut, N, P);
    MatrixPrint(MatOut, N, P);

    //Fin du premier travail dans le CPU (3 fonctions)

    //Début du premier travail dans le GPU

    cudaMalloc((void**)&d_Mat1, sizeof(float)*(N*P));
    cudaMalloc((void**)&d_Mat2, sizeof(float)*(N*P));
    cudaMalloc((void**)&d_MatOut, sizeof(float)*(N*P));

    cudaMemcpy(d_Mat1, Mat1, sizeof(float) * (N*P), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mat2, Mat2, sizeof(float) * (N*P), cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatOut, MatOut, sizeof(float) * (N*P), cudaMemcpyHostToDevice);

    cudaMatrixAdd<<<N,P>>>(d_Mat1, d_Mat2, d_MatOut);

    cudaMemcpy(MatOut, d_MatOut, sizeof(float) * (N*P), cudaMemcpyDeviceToHost);

    MatrixPrint(MatOut, N, P);

    //Fin du premier travail dans le GPU

    //Début du deuxième travail dans le CPU

    MatrixMult(Mat1, Mat2, MatOut, N);
    MatrixPrint(MatOut, N, N);

    //Fin du deuxième travail dans le CPU

    //Début du deuxième travail dans le GPU

    cudaMatrixMult<<<N,N>>>(d_Mat1,d_Mat2,d_MatOut);

    cudaMemcpy(MatOut, d_MatOut, sizeof(float) * (N*N), cudaMemcpyDeviceToHost);

    MatrixPrint(MatOut, N, N);
    //Fin du deuxième travail dans le GPU

    cudaFree(d_Mat1);
    cudaFree(d_Mat2);
    cudaFree(d_MatOut);

    free(Mat1);
    free(Mat2);
    free(MatOut);

    cudaDeviceSynchronize();
    return 0 ;
}
