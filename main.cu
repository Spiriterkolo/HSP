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

int main() {
    cuda_hello<<<1,1>>>();
    int N = 3;
    int P = 2;
    float* Mat1;
    float* Mat2;

    Mat1 = (float*)malloc(sizeof(float) * (N*P));
    Mat2 = (float*)malloc(sizeof(float) * (N*P));

    MatrixInit(Mat1, N, P);
    MatrixInit(Mat2, N, P);

    MatrixPrint(Mat1, N, P);
    MatrixPrint(Mat2, N, P);

    float* MatOut;
    MatOut = (float*)malloc(sizeof(float) * (N*P));

    MatrixAdd(Mat1, Mat2, MatOut, N, P);

    MatrixPrint(MatOut, N, P);

    free(Mat1);
    free(Mat2);
    free(MatOut);
    //Fin du travail dans le GPU

    //Début du travail dans le CPU

    



    cudaDeviceSynchronize();
    return 0 ;
}
