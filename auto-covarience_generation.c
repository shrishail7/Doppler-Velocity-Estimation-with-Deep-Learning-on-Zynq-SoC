#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>

#define M_PI 3.14159265358979323846 
#define M 20  // Total Input Length
#define L 10  // Subarray Length
#define K (M - L + 1)  // 11 subarrays

double complex x[M];           // Input signal vector
double complex R_avg[L][L];    // Output: Averaged autocovariance matrix
double complex Smoothing_vector[K][L];


void compute_autocovariance_spatial_smoothing(double complex *input) {
    int i, j, n;
    
    // Zero initialize
    memset(R_avg, 0, sizeof(R_avg));
    memset(Smoothing_vector,0,sizeof(Smoothing_vector));
    // Compute average over K overlapping subarrays
    for (n = 0; n < K; n++) {
        for (i = 0; i < L; i++) {
            Smoothing_vector[n][i] = input[n+i];
            for (j = 0; j < L; j++) {
                // R[i][j] += conj(x[n+i]) * x[n+j] / K
                
                R_avg[i][j] += conj(input[n + i]) * input[n + j];
            }
        }
    }
    
    // Normalize by #subarrays
    for (i = 0; i < L; i++) {
        for (j = 0; j < L; j++) {
            R_avg[i][j] /= K;
        }
    }
    
}

int main() {
    int i, j;
    
    for (int i = 0; i < M; i++) {
        x[i] = (i + 1) + (i * 0.5) * I;
    }

    printf("\n");
    printf("****************************************************************************************\n");
    printf("****************************************************************************************\n");
    printf("                                    Input signal                                    \n");
    printf("****************************************************************************************\n");
    printf("****************************************************************************************\n");

    

    for (i = 0; i < M; i++) {
        printf("x[%d] = %.3f + %.3fi\n", i, creal(x[i]), cimag(x[i]));
    }
    
    // Compute autocovariance
    compute_autocovariance_spatial_smoothing(x);
    
    printf("\n");
    printf("****************************************************************************************\n");
    printf("****************************************************************************************\n");
    printf("                                    Smoothing vectors                                    \n");
    printf("****************************************************************************************\n");
    printf("****************************************************************************************\n");

    for(int i=0;i<L;i++){
        printf("x[%d] = ",i);
        for(int j=0;j<L;j++){
            printf(" ( %6.3f + %6.3f i ) ", creal(Smoothing_vector[i][j]) , cimag(Smoothing_vector[i][j]));
        }
        printf("\n");
    }


    printf("\n");
    printf("****************************************************************************************\n");
    printf("****************************************************************************************\n");
    printf("                                    Autocovariance R_avg                                   \n");
    printf("****************************************************************************************\n");
    printf("****************************************************************************************\n");
    for (i = 0; i < L; i++) {
        for (j = 0; j < L; j++) {
            printf("(%6.3f,%6.3f) ", creal(R_avg[i][j]), cimag(R_avg[i][j]));
        }
        printf("\n");
    }
    
    printf("\nDiagonal (variances): ");
    for (i = 0; i < L; i++) {
        printf("%.3f ", creal(R_avg[i][i]));
    }
    printf("\n");
    
    return 0;
}
