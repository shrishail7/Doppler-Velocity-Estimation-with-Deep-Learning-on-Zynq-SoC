#include <stdio.h>
#include <complex.h>

#define M 20  // Total Input Length
#define L 10  // Subarray Length (Window)
#define K (M - L + 1) // Number of subarrays (11)

void spatial_smoothing(double complex *input, double complex output_cov[L][L]) {
    // Initialize output matrix to zero
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            output_cov[i][j] = 0;
        }
    }


    for(int i=0;i<M-L+1;i++){
        for(int j=0;j<L;j++){
            output_cov[i][j]=input[i+j];
        }
    }

}

int main() {
    double complex signal[M];
    double complex smoothed_R[L][L];

    // Dummy data: 20 complex inputs
    for (int i = 0; i < M; i++) {
        signal[i] = (i + 1) + (i * 0.5) * I;
    }

    spatial_smoothing(signal, smoothed_R);

    for(int i=0;i<M;i++){
        printf("%f + %f i \n",creal(signal[i]) , cimag(signal[i]));
    }

    // Print a small part of the resulting matrix
    printf("Smoothed Covariance Matrix :\n");
    for (int i = 0; i < L; i++) {
        printf("%d th vector =",i);
        for (int j = 0; j < L; j++) {
            printf("(%.2f + %.2fi) ", creal(smoothed_R[i][j]), cimag(smoothed_R[i][j]));
        }
        printf("\n");
    }

    return 0;
}