#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <xtime_l.h>
/*
 * ESPRIT Algorithm Implementation
 * Developed in alignment with original logic from main.ipynb
 */

// ==========================================
// Data Structures
// ==========================================
typedef struct {
    float real;
    float imag;
} Complex;

// Include the generated dataset header
// Note: This must come AFTER the Complex struct definition
#include "inputs.h"

// ==========================================
// Helper Functions for Complex Arithmetic
// ==========================================

void print_complex(Complex c) {
    if (c.imag >= 0)
        printf("%.10f + %.10fi", c.real, c.imag);
    else
        printf("%.10f - %.10fi", c.real, -c.imag);
}

Complex add_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

Complex sub_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

Complex mul_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

Complex mul_complex_scalar(Complex a, float s) {
    Complex result;
    result.real = a.real * s;
    result.imag = a.imag * s;
    return result;
}

Complex div_complex_scalar(Complex a, float s) {
    Complex result;
    result.real = a.real / s;
    result.imag = a.imag / s;
    return result;
}

Complex conj_complex(Complex a) {
    Complex result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

// -------------------------------------------------------------
// Matrix Operations Helpers
// -------------------------------------------------------------

// Matrix multiplication: C = A * B (All matrices are n x n)
// Logic equivalent to Python: A @ B
void matmul_complex(Complex *A, Complex *B, Complex *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Complex sum = {0.0f, 0.0f};
            for (int k = 0; k < n; k++) {
                sum = add_complex(sum, mul_complex(A[i*n+k], B[k*n+j]));
            }
            C[i*n+j] = sum;
        }
    }
}

// Transpose and Conjugate Matrix: B = A^H
// Equivalent to Python: np.transpose(np.conj(A))
void transpose_conj_matrix(Complex *A, Complex *B, int n) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            B[j*n+i] = conj_complex(A[i*n+j]);
        }
    }
}

// Copy Matrix: dst = src
// Equivalent to Python: A.copy()
void copy_matrix(Complex *src, Complex *dst, int n) {
    for(int i=0; i<n*n; i++) {
        dst[i] = src[i];
    }
}

// Identity Matrix: A = I
// Equivalent to Python: np.identity(m, dtype=np.complex64)
void identity_matrix(Complex *A, int n) {
    for(int i=0; i<n*n; i++) {
        A[i].real = 0.0f; A[i].imag = 0.0f;
    }
    for(int i=0; i<n; i++) {
        A[i*n+i].real = 1.0f;
    }
}

// ==========================================
// ESPRIT Core Functions
// ==========================================

/*
 * Python equivalent:
 * def autocorrelation_func_ES(rec_sig):
 *     conj_sig = np.conj(rec_sig)
 *     autocorrelation_matrix = []
 *     for i in range(len(conj_sig)):
 *         dat1 = conj_sig[i] * rec_sig
 *         autocorrelation_matrix.append(dat1)
 */
void autocorrelation_func_ES(Complex *rec_sig, int len, Complex *autocorrelation_matrix) {
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
        	Complex val = mul_complex(conj_complex(rec_sig[i]), rec_sig[j]);
            autocorrelation_matrix[i * len + j] = val;
        }
    }
}

/*
 * Logic taken from ESPRIT_PS_S_PINV internal loop:
 * for i in range(subarr):
 *     sub1 = rec_signal[i:(l + i)]
 *     sub2 = autocorrelation_func_ES(np.array(sub1))
 *     autocorrelation_matrix += sub2
 * autocorrelation_matrix /= subarr
 */
void spatial_smoothing(const Complex *rec_signal, int N, Complex *output_matrix) {
    int l = N / 2; // length of subarray
    int subarr = N + 1 - l; // number of subarrays

    // Initialize output matrix to 0
    for(int i=0; i<l*l; i++) {
        output_matrix[i].real = 0.0f; output_matrix[i].imag = 0.0f;
    }

    // Allocate temporary buffers
    Complex *sub_auto_mat = (Complex *)malloc(l * l * sizeof(Complex));
    Complex *sub_signal = (Complex *)malloc(l * sizeof(Complex));

    for (int i = 0; i < subarr; i++) {
        // Extract subarray (Equivalent to rec_signal[i:(l + i)])
        for(int j=0; j<l; j++) {
            sub_signal[j] = rec_signal[i + j];
        }

        autocorrelation_func_ES(sub_signal, l, sub_auto_mat);

        // Accumulate (Equivalent to += sub2)
        for(int k=0; k<l*l; k++) {
            output_matrix[k] = add_complex(output_matrix[k], sub_auto_mat[k]);
        }
    }

    // Normalize (Equivalent to /= subarr)
    for(int k=0; k<l*l; k++) {
        output_matrix[k] = div_complex_scalar(output_matrix[k], (float)subarr);
    }

    free(sub_auto_mat);
    free(sub_signal);
}

/*
 * Python equivalent:
 * def givensrotation_new(a, b):
 *     hypo=np.sqrt(a.real**2 + a.imag**2 + b.real**2 + b.imag**2)
 */
void givensrotation_new(Complex a, Complex b, Complex *cos_out, Complex *sin_out) {
    float hypo = sqrtf(a.real*a.real + a.imag*a.imag + b.real*b.real + b.imag*b.imag);
    if (hypo == 0) {
        cos_out->real = 1.0f; cos_out->imag = 0.0f;
        sin_out->real = 0.0f; sin_out->imag = 0.0f;
    } else {
        *cos_out = div_complex_scalar(a, hypo);
        *sin_out = div_complex_scalar(b, hypo);
    }
}

/*
 * Python equivalent:
 * def qr_givens_S(A):
 *     ... updates R and Q using rows ...
 *     return np.transpose(np.conj(Q)), R
 */
void qr_givens_S(Complex *A, int n, Complex *Q_out, Complex *R_out) {
    copy_matrix(A, R_out, n);
    Complex *Q_temp = (Complex *)malloc(n * n * sizeof(Complex));
    identity_matrix(Q_temp, n);

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            Complex cos, sin;
            givensrotation_new(R_out[i*n+i], R_out[j*n+i], &cos, &sin);

            Complex conj_cos = conj_complex(cos);
            Complex conj_sin = conj_complex(sin);
            Complex neg_sin = mul_complex_scalar(sin, -1.0f);

            // Apply rotations to selected rows of Q and R
            for (int k = 0; k < n; k++) {
                Complex R_i = R_out[i*n+k];
                Complex R_j = R_out[j*n+k];
                R_out[i*n+k] = add_complex(mul_complex(R_i, conj_cos), mul_complex(R_j, conj_sin));
                R_out[j*n+k] = add_complex(mul_complex(R_i, neg_sin), mul_complex(R_j, cos));

                Complex Q_i = Q_temp[i*n+k];
                Complex Q_j = Q_temp[j*n+k];
                Q_temp[i*n+k] = add_complex(mul_complex(Q_i, conj_cos), mul_complex(Q_j, conj_sin));
                Q_temp[j*n+k] = add_complex(mul_complex(Q_i, neg_sin), mul_complex(Q_j, cos));
            }
        }
    }
    transpose_conj_matrix(Q_temp, Q_out, n);
    free(Q_temp);
}

/*
 * Python equivalent:
 * def qr_algorithm_eig_single(A_input, max_iterations=20):
 *     ... A = R @ Q, Q_total = Q_total @ Q ...
 */
void qr_algorithm_eig_single(Complex *A_input, int n, int max_iterations, Complex *eigenvalues, Complex *eigenvectors) {
    Complex *A = (Complex *)malloc(n * n * sizeof(Complex));
    copy_matrix(A_input, A, n);
    identity_matrix(eigenvectors, n);

    Complex *Q = (Complex *)malloc(n * n * sizeof(Complex));
    Complex *R = (Complex *)malloc(n * n * sizeof(Complex));
    Complex *TempA = (Complex *)malloc(n * n * sizeof(Complex));
    Complex *TempQTotal = (Complex *)malloc(n * n * sizeof(Complex));

    for (int iter = 0; iter < max_iterations; iter++) {
        qr_givens_S(A, n, Q, R);
        matmul_complex(R, Q, TempA, n);
        copy_matrix(TempA, A, n);
        matmul_complex(eigenvectors, Q, TempQTotal, n);
        copy_matrix(TempQTotal, eigenvectors, n);
    }

    // np.diag(A)
    for(int i=0; i<n; i++) {
        eigenvalues[i] = A[i*n+i];
    }

    free(A); free(Q); free(R); free(TempA); free(TempQTotal);
}

// Analytical inverse for 2x2: (1/det) * adj(A)
void inverse_2x2_complex(Complex *A, Complex *invA) {
    Complex det = sub_complex(mul_complex(A[0], A[3]), mul_complex(A[1], A[2]));
    float det_norm_sq = det.real*det.real + det.imag*det.imag;
    Complex invDet = {det.real / det_norm_sq, -det.imag / det_norm_sq};

    invA[0] = mul_complex(invDet, A[3]);
    invA[1] = mul_complex(invDet, mul_complex_scalar(A[1], -1.0f));
    invA[2] = mul_complex(invDet, mul_complex_scalar(A[2], -1.0f));
    invA[3] = mul_complex(invDet, A[0]);
}

/*
 * Logic equivalent to Python:
 * phi_mat = np.linalg.pinv(subA1) @ subB1
 * Implementation: pinv(A) = inv(A^H * A) * A^H
 */
void pinv_complex_k2(Complex *A, int m, Complex *pinvA) {
    Complex *AH = (Complex *)malloc(2 * m * sizeof(Complex));
    for(int i=0; i<m; i++) {
        for(int j=0; j<2; j++) {
            AH[j*m+i] = conj_complex(A[i*2+j]);
        }
    }
    Complex G[4];
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            Complex sum = {0.0f, 0.0f};
            for(int k=0; k<m; k++) {
                sum = add_complex(sum, mul_complex(AH[i*m+k], A[k*2+j]));
            }
            G[i*2+j] = sum;
        }
    }
    Complex invG[4];
    inverse_2x2_complex(G, invG);
    for(int i=0; i<2; i++) {
        for(int j=0; j<m; j++) {
            Complex sum = {0.0f, 0.0f};
            for(int k=0; k<2; k++) {
                sum = add_complex(sum, mul_complex(invG[i*2+k], AH[k*m+j]));
            }
            pinvA[i*m+j] = sum;
        }
    }
    free(AH);
}

/*
 * Python equivalent:
 * def eign_cal_ES(phi_mat):
 *     trace = sum(diag), det = ad-bc
 *     lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
 */
void eign_cal_ES(Complex *phi_mat, Complex *eigenvalues) {
    Complex trace = add_complex(phi_mat[0], phi_mat[3]);
    Complex det = sub_complex(mul_complex(phi_mat[0], phi_mat[3]), mul_complex(phi_mat[1], phi_mat[2]));
    Complex disc = sub_complex(mul_complex(trace, trace), mul_complex_scalar(det, 4.0f));
    float r = sqrtf(disc.real*disc.real + disc.imag*disc.imag);
    float theta = atan2f(disc.imag, disc.real);
    Complex sqrt_disc = {sqrtf(r) * cosf(theta / 2.0f), sqrtf(r) * sinf(theta / 2.0f)};
    eigenvalues[0] = div_complex_scalar(add_complex(trace, sqrt_disc), 2.0f);
    eigenvalues[1] = div_complex_scalar(sub_complex(trace, sqrt_disc), 2.0f);
}

// Ensures signal subspace corresponds to most dominant components
void sort_eigenvalues(Complex *eigenvalues, Complex *eigenvectors, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            float mag1 = sqrtf(
    eigenvalues[j].real*eigenvalues[j].real +
    eigenvalues[j].imag*eigenvalues[j].imag
);

float mag2 = sqrtf(
    eigenvalues[j+1].real*eigenvalues[j+1].real +
    eigenvalues[j+1].imag*eigenvalues[j+1].imag
);
            if (mag1 < mag2) {
                Complex tV = eigenvalues[j]; eigenvalues[j] = eigenvalues[j+1]; eigenvalues[j+1] = tV;
                for (int r = 0; r < n; r++) {
                    Complex tP = eigenvectors[r*n+j]; eigenvectors[r*n+j] = eigenvectors[r*n+j+1]; eigenvectors[r*n+j+1] = tP;
                }
            }
        }
    }
}

// Python equivalent: np.clip(vels, -30, 30)
float vel_clip(float v) {
    if (v < -30.0f) return -30.0f;
    if (v > 30.0f) return 30.0f;
    return v;
}

 int main() {
     printf("========================================\n");
     printf("   ESPRIT BATCH PROCESSING (15 SAMPLES) \n");
     printf("   (PRECISION: FLOAT)                   \n");
     printf("========================================\n");

 	// XTime esprit_start, esprit_end;
 	// float total_time;

 	// XTime_SetTime(0);
 	// XTime_GetTime(&esprit_start);

     /*
      * Batch processing logic mirrors Jupyter cell 9:
      * for SNR in snr_vals:
      *     for input_tensor, label_tensor in tqdm(test_dataset):
      */
     // Variables for RMSE calculation
     float snr_sq_err[3] = {0.0f, 0.0f, 0.0f};
     int snr_processed_count[3] = {0, 0, 0};
     int snr_list[3] = {-20, 0, 20};

     for (int s = 0; s < NUM_SAMPLES; s++) {
         ESPRIT_Sample current = dataset[s];
         int N = N_SAMPLES;
         int k = 2; // Signal subspace dimension (M in Python)
         int l = N / 2; // Subarray length (L in Python)

         printf("\n>>> PROCESSING: SNR %d, Sample ID %d <<<\n", current.snr, current.sample_id);
         printf("Ground Truth: %.10f, %.10f\n", current.gt_vel1, current.gt_vel2);

         // 1. Spatial Smoothing (Cell 6 logic)
         Complex *autocorrelation_matrix = (Complex *)malloc(l * l * sizeof(Complex));
         spatial_smoothing(current.data, N, autocorrelation_matrix);

         // 2. Eigenvalue Decomposition (Cell 6: qr_algorithm_eig_single)
         Complex *eigenvalues_all = (Complex *)malloc(l * sizeof(Complex));
         Complex *eigenvectors_all = (Complex *)malloc(l * l * sizeof(Complex));
         qr_algorithm_eig_single(autocorrelation_matrix, l, 20, eigenvalues_all, eigenvectors_all);

         // 3. Sorting (Implicit in Notebook, Explicit here for stability)
         sort_eigenvalues(eigenvalues_all, eigenvectors_all, l);

         // 4. Sub-matrices (Cell 6: Signal subspace S extract and split)
         // S = eigenvectors[:, :k]
         // subA1 = S[0:L-1, :], subB1 = S[1:L, :]
         int rows_sub = l - 1;
         Complex *subA1 = (Complex *)malloc(rows_sub * k * sizeof(Complex));
         Complex *subB1 = (Complex *)malloc(rows_sub * k * sizeof(Complex));
         for(int j=0; j<k; j++) {
             for(int i=0; i<rows_sub; i++) {
                 subA1[i*k+j] = eigenvectors_all[i*l+j];
                 subB1[i*k+j] = eigenvectors_all[(i+1)*l+j];
             }
         }

         // 5. Phi Matrix (Cell 6: phi_mat = pinv(subA1) @ subB1)
         Complex *pinvA1 = (Complex *)malloc(k * rows_sub * sizeof(Complex));
         pinv_complex_k2(subA1, rows_sub, pinvA1);
         Complex phi_mat[4];
         for(int i=0; i<k; i++) {
             for(int j=0; j<k; j++) {
                 Complex sum = {0.0f, 0.0f};
                 for(int m=0; m<rows_sub; m++) {
                     sum = add_complex(sum, mul_complex(pinvA1[i*rows_sub+m], subB1[m*k+j]));
                 }
                 phi_mat[i*k+j] = sum;
             }
         }

         // 6. Eigenvalues of Phi & Doppler estimation (Cell 6: phi = eign_cal_ES(phi_mat))
         // omega_estimates = -np.angle(phi) / 0.005
         Complex phi_eigs[2];
         eign_cal_ES(phi_mat, phi_eigs);
         float estimates[2];
         for(int i=0; i<k; i++) {
             float angle = atan2f(phi_eigs[i].imag, phi_eigs[i].real);
             estimates[i] = -angle / 0.005f;
         }

         // Sort velocities in ascending order
         if (estimates[0] > estimates[1]) {
             float temp = estimates[0];
             estimates[0] = estimates[1];
             estimates[1] = temp;
         }

         float clipped0 = vel_clip(estimates[0]);
         float clipped1 = vel_clip(estimates[1]);
         printf("Estimates:    %.10f, %.10f m/s\n", clipped0, clipped1);

         // --- RMSE Accumulation ---
         // Sort Ground Truth for fair matching with sorted estimates
         float gt[2] = {(float)current.gt_vel1, (float)current.gt_vel2};
         if (gt[0] > gt[1]) {
             float temp = gt[0];
             gt[0] = gt[1];
             gt[1] = temp;
         }

         // Identify SNR index
         int snr_idx = -1;
         for(int i=0; i<3; i++) {
             if(current.snr == snr_list[i]) {
                 snr_idx = i;
                 break;
             }
         }

         if(snr_idx != -1) {
             snr_sq_err[snr_idx] += powf(clipped0 - gt[0], 2.0f);
             snr_sq_err[snr_idx] += powf(clipped1 - gt[1], 2.0f);
             snr_processed_count[snr_idx]++;
         }

         // Cleanup
         free(autocorrelation_matrix);
         free(eigenvalues_all);
         free(eigenvectors_all);
         free(subA1);
         free(subB1);
         free(pinvA1);
     }

 	// XTime_GetTime(&esprit_end);



 	// total_time = (float)1.0 * (esprit_end -esprit_start) / (COUNTS_PER_SECOND / 1000000);
 	// printf("\n######## Execution Time ####\n");
 	// printf("PS Execution Time for ESPRIT (micro-sec) %f \n", total_time/15);


     printf("\n========================================\n");
     printf("   RMSE RESULTS FOR EACH SNR\n");
     printf("========================================\n");
     for(int i=0; i<3; i++) {
         if(snr_processed_count[i] > 0) {
             float rmse = sqrtf(snr_sq_err[i] / (snr_processed_count[i] * 2.0f));
             printf("SNR %4d: RMSE = %.10f\n", snr_list[i], rmse);
         }
     }

     printf("\n========================================\n");
     printf("   BATCH PROCESSING COMPLETE            \n");
     printf("========================================\n");

     return 0;
 }
