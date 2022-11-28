#include <stdio.h>
#include <stdlib.h>

void MatrixInit(float *M, int n, int p){
    float max = RAND_MAX;
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[j + i*p] = (rand() / max) * 2 - 1.0 ;
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    printf("Matrice : ");
    for(int i=0; i<n; i++){
        printf("\n");
        for(int j=0; j<p; j++){
            printf("%f ", M[j + i*p]);
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
        for (int j = 0; i < blockDim.x; j++){
            Mout[j + i * blockDim.x] = M1[j + i * blockDim.x] + M2[j + i * blockDim.x];
        }
    }
}

int main(){
    int n = 4;
    int p = 3;
    //float *M;
    //M = (float*)malloc((sizeof (float))* n * p);
    //MatrixInit(M, n, p);
    //MatrixPrint(M, n, p);
    //free(M);
    float *M1, *M2, *Mout;
    M1 = (float*)malloc((sizeof (float))* n * p);
    M2 = (float*)malloc((sizeof (float))* n * p);
    Mout = (float*)malloc((sizeof (float))* n * p);
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
    MatrixPrint(M1, n, p);
    MatrixPrint(M2, n, p);
    MatrixAdd(M1, M2, Mout, n, p);
    MatrixPrint(Mout, n, p);
    free(M1);
    free(M2);
    free(Mout);
    return 0;
}