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
}

int main(){
    int n = 4;
    int p = 3;
    float *M;
    M = (float*)malloc((sizeof (float))* n * p);
    MatrixInit(M, n, p);
    MatrixPrint(M, n, p);
    free(M);
    return 0;
}