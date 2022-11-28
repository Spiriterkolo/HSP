#include <stdio.h>
#include <stdlib.h>

void MatrixInit(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            M[i][j] = (float)(rand()/RAND_MAX)*2-1;
        }
    }
}

int main(){
    return 0;
}