#include "esprit.h"
#include <math.h>
#include "inputs.h"

// --- Benchmark Helper Functions ---
static inline Complex bm_conj_complex(Complex a) {
    Complex result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

static void bm_matmul_complex(const Complex *A, const Complex *B, Complex *C, int n) {
    for (int i = 0; i < n; i++) {
        int i_n = i * n;
        for (int j = 0; j < n; j++) {
            float sum_r = 0.0f, sum_i = 0.0f;
            for (int k = 0; k < n; k++) {
                int k_n = k * n;
                float ar = A[i_n + k].real, ai = A[i_n + k].imag;
                float br = B[k_n + j].real, bi = B[k_n + j].imag;
                sum_r += ar * br - ai * bi;
                sum_i += ar * bi + ai * br;
            }
            C[i_n + j].real = sum_r;
            C[i_n + j].imag = sum_i;
        }
    }
}

static void bm_transpose_conj_matrix(Complex *A, Complex *B, int n) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            B[j*n+i] = bm_conj_complex(A[i*n+j]);
        }
    }
}

static void bm_copy_matrix(Complex *src, Complex *dst, int n) {
    for(int i=0; i<n*n; i++) {
        dst[i] = src[i];
    }
}

static void bm_identity_matrix(Complex *A, int n) {
    for(int i=0; i<n*n; i++) {
        A[i].real = 0.0f; A[i].imag = 0.0f;
    }
    for(int i=0; i<n; i++) {
        A[i*n+i].real = 1.0f;
    }
}

static void bm_spatial_smoothing(const Complex *rec_signal, int N, Complex *output_matrix) {
    int l = N / 2;
    int subarr = N + 1 - l;

    float accum_r[100] = {0};
    float accum_i[100] = {0};

    for (int i = 0; i < subarr; i++) {
        const Complex *sig_ptr = &rec_signal[i];
        for(int j=0; j<l; j++) {
            int j_l = j * l;
            float r_j_r = sig_ptr[j].real;
            float r_j_i = sig_ptr[j].imag;

            for(int k=0; k<l; k++) {
                float r_k_r = sig_ptr[k].real;
                float r_k_i = sig_ptr[k].imag;

                accum_r[j_l + k] += r_j_r * r_k_r + r_j_i * r_k_i;
                accum_i[j_l + k] += r_j_r * r_k_i - r_j_i * r_k_r;
            }
        }
    }

    float inv_subarr = 1.0f / subarr;

    for(int k=0; k<l*l; k++) {
        output_matrix[k].real = (accum_r[k] * inv_subarr);
        output_matrix[k].imag = (accum_i[k] * inv_subarr);
    }
}

static void bm_givensrotation_new(Complex a, Complex b, Complex *cos_out, Complex *sin_out) {
    float a_r = a.real, a_i = a.imag;
    float b_r = b.real, b_i = b.imag;

    float hypo = sqrtf(a_r*a_r + a_i*a_i + b_r*b_r + b_i*b_i);

    if (hypo == 0.0f) {
        cos_out->real = 1.0f; cos_out->imag = 0.0f;
        sin_out->real = 0.0f; sin_out->imag = 0.0f;
    } else {
        cos_out->real = (a_r / hypo);
        cos_out->imag = (a_i / hypo);

        sin_out->real = (b_r / hypo);
        sin_out->imag = (b_i / hypo);
    }
}

static void bm_qr_givens_S(const Complex *A, int n, Complex *Q_out, Complex *R_out) {
    bm_copy_matrix((Complex *)A, R_out, n);
    Complex Q_temp[100];
    bm_identity_matrix(Q_temp, n);

    for (int i = 0; i < n - 1; i++) {
        int i_n = i * n;
        for (int j = i + 1; j < n; j++) {
            int j_n = j * n;
            Complex cosf, sinf;
            bm_givensrotation_new(R_out[i_n + i], R_out[j_n + i], &cosf, &sinf);

            float c_r = cosf.real, c_i = cosf.imag;
            float s_r = sinf.real, s_i = sinf.imag;

            float conj_c_r = c_r, conj_c_i = -c_i;
            float conj_s_r = s_r, conj_s_i = -s_i;
            float neg_s_r = -s_r, neg_s_i = -s_i;

            for (int k = 0; k < n; k++) {
                float Ri_r = R_out[i_n + k].real, Ri_i = R_out[i_n + k].imag;
                float Rj_r = R_out[j_n + k].real, Rj_i = R_out[j_n + k].imag;

                float t1_r = Ri_r*conj_c_r - Ri_i*conj_c_i + Rj_r*conj_s_r - Rj_i*conj_s_i;
                float t1_i = Ri_r*conj_c_i + Ri_i*conj_c_r + Rj_r*conj_s_i + Rj_i*conj_s_r;

                float t2_r = Ri_r*neg_s_r - Ri_i*neg_s_i + Rj_r*c_r - Rj_i*c_i;
                float t2_i = Ri_r*neg_s_i + Ri_i*neg_s_r + Rj_r*c_i + Rj_i*c_r;

                R_out[i_n + k].real = t1_r;
                R_out[i_n + k].imag = t1_i;

                R_out[j_n + k].real = t2_r;
                R_out[j_n + k].imag = t2_i;

                float Qi_r = Q_temp[i_n + k].real, Qi_i = Q_temp[i_n + k].imag;
                float Qj_r = Q_temp[j_n + k].real, Qj_i = Q_temp[j_n + k].imag;

                float qt1_r = Qi_r*conj_c_r - Qi_i*conj_c_i + Qj_r*conj_s_r - Qj_i*conj_s_i;
                float qt1_i = Qi_r*conj_c_i + Qi_i*conj_c_r + Qj_r*conj_s_i + Qj_i*conj_s_r;

                float qt2_r = Qi_r*neg_s_r - Qi_i*neg_s_i + Qj_r*c_r - Qj_i*c_i;
                float qt2_i = Qi_r*neg_s_i + Qi_i*neg_s_r + Qj_r*c_i + Qj_i*c_r;

                Q_temp[i_n + k].real = qt1_r;
                Q_temp[i_n + k].imag = qt1_i;

                Q_temp[j_n + k].real = qt2_r;
                Q_temp[j_n + k].imag = qt2_i;
            }
        }
    }
    bm_transpose_conj_matrix(Q_temp, Q_out, n);
}

static void bm_qr_algorithm_eig_single(const Complex *A_input, int n, int max_iterations, Complex *eigenvalues, Complex *eigenvectors) {
    Complex A[100], Q[100], R[100], TempA[100], TempQTotal[100];
    bm_copy_matrix((Complex *)A_input, A, n);
    bm_identity_matrix(eigenvectors, n);

    for (int iter = 0; iter < max_iterations; iter++) {
        bm_qr_givens_S(A, n, Q, R);
        bm_matmul_complex(R, Q, TempA, n);
        bm_copy_matrix(TempA, A, n);

        bm_matmul_complex(eigenvectors, Q, TempQTotal, n);
        bm_copy_matrix(TempQTotal, eigenvectors, n);
    }
    for(int i=0; i<n; i++) {
        eigenvalues[i] = A[i*n+i];
    }
}

static void bm_svd_pinv_complex_k2(Complex *A, int m, Complex *pinvA) {
    float H[2][2][2] ;
    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            float sum_r = 0, sum_i = 0;
            for(int k=0; k<m; ++k) {
                float a_ik_r = A[k*2+i].real, a_ik_i = -A[k*2+i].imag;
                float a_kj_r = A[k*2+j].real, a_kj_i = A[k*2+j].imag;
                sum_r += a_ik_r * a_kj_r - a_ik_i * a_kj_i;
                sum_i += a_ik_r * a_kj_i + a_ik_i * a_kj_r;
            }
            H[i][j][0] = sum_r;
            H[i][j][1] = sum_i;
        }
    }

    float trace_r = H[0][0][0] + H[1][1][0];
    float trace_i = H[0][0][1] + H[1][1][1];
    float ad_r = H[0][0][0] * H[1][1][0] - H[0][0][1] * H[1][1][1];
    float ad_i = H[0][0][0] * H[1][1][1] + H[0][0][1] * H[1][1][0];
    float bc_r = H[0][1][0] * H[1][0][0] - H[0][1][1] * H[1][0][1];
    float bc_i = H[0][1][0] * H[1][0][1] + H[0][1][1] * H[1][0][0];

    float det_r = ad_r - bc_r;
    float det_i = ad_i - bc_i;

    float tr_sq_r = trace_r*trace_r - trace_i*trace_i;
    float tr_sq_i = 2*trace_r*trace_i;
    float disc_r = tr_sq_r - 4*det_r;
    float disc_i = tr_sq_i - 4*det_i;

    float disc_mag = sqrtf(disc_r*disc_r + disc_i*disc_i);
    float disc_ang = atan2f(disc_i, disc_r);

    float sqrt_disc_r = sqrtf(disc_mag) * cosf(disc_ang / 2.0f);
    float sqrt_disc_i = sqrtf(disc_mag) * sinf(disc_ang / 2.0f);

    float lambda1_r = (trace_r + sqrt_disc_r) / 2.0f;
    float lambda1_i = (trace_i + sqrt_disc_i) / 2.0f;
    float lambda2_r = (trace_r - sqrt_disc_r) / 2.0f;
    float lambda2_i = (trace_i - sqrt_disc_i) / 2.0f;

    float l1 = sqrtf(lambda1_r*lambda1_r + lambda1_i*lambda1_i);
    float l2 = sqrtf(lambda2_r*lambda2_r + lambda2_i*lambda2_i);

    float tol = 1e-12;
    float inv_l1 = (l1 > tol) ? 1.0f / l1 : 0.0f;
    float inv_l2 = (l2 > tol) ? 1.0f / l2 : 0.0f;

    float V[2][2][2] = {{{1, 0}, {0, 0}}, {{0, 0}, {1, 0}}};
    float h11_l1_r = H[0][0][0] - lambda1_r;
    float h11_l1_i = H[0][0][1] - lambda1_i;

    if (H[0][1][0]*H[0][1][0] + H[0][1][1]*H[0][1][1] > 1e-20) {
        float mag1 = sqrtf(H[0][1][0]*H[0][1][0] + H[0][1][1]*H[0][1][1] + h11_l1_r*h11_l1_r + h11_l1_i*h11_l1_i);
        V[0][0][0] = H[0][1][0]/mag1; V[0][0][1] = H[0][1][1]/mag1;
        V[1][0][0] = -h11_l1_r/mag1; V[1][0][1] = -h11_l1_i/mag1;
    }

    float h11_l2_r = H[0][0][0] - lambda2_r;
    float h11_l2_i = H[0][0][1] - lambda2_i;
    if (H[0][1][0]*H[0][1][0] + H[0][1][1]*H[0][1][1] > 1e-20) {
        float mag2 = sqrtf(H[0][1][0]*H[0][1][0] + H[0][1][1]*H[0][1][1] + h11_l2_r*h11_l2_r + h11_l2_i*h11_l2_i);
        V[0][1][0] = H[0][1][0]/mag2; V[0][1][1] = H[0][1][1]/mag2;
        V[1][1][0] = -h11_l2_r/mag2; V[1][1][1] = -h11_l2_i/mag2;
    }

    float H_inv[2][2][2];
    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            float sum_r = 0, sum_i = 0;
            for(int k=0; k<2; ++k) {
                float inv_lk = (k == 0) ? inv_l1 : inv_l2;
                float vik_r = V[i][k][0] * inv_lk, vik_i = V[i][k][1] * inv_lk;
                float vhkj_r = V[j][k][0], vhkj_i = -V[j][k][1];
                sum_r += vik_r * vhkj_r - vik_i * vhkj_i;
                sum_i += vik_r * vhkj_i + vik_i * vhkj_r;
            }
            H_inv[i][j][0] = sum_r; H_inv[i][j][1] = sum_i;
        }
    }

    for(int i=0; i<2; ++i) {
        for(int j=0; j<m; ++j) {
            float sum_r = 0, sum_i = 0;
            for(int k=0; k<2; ++k) {
                float ah_kj_r = A[j*2+k].real, ah_kj_i = -A[j*2+k].imag;
                sum_r += H_inv[i][k][0] * ah_kj_r - H_inv[i][k][1] * ah_kj_i;
                sum_i += H_inv[i][k][0] * ah_kj_i + H_inv[i][k][1] * ah_kj_r;
            }
            pinvA[i*m+j].real = sum_r;
            pinvA[i*m+j].imag = sum_i;
        }
    }
}

static void bm_eign_cal_ES(Complex *phi_mat, Complex *eigenvalues) {
    float r00 = phi_mat[0].real, i00 = phi_mat[0].imag;
    float r01 = phi_mat[1].real, i01 = phi_mat[1].imag;
    float r10 = phi_mat[2].real, i10 = phi_mat[2].imag;
    float r11 = phi_mat[3].real, i11 = phi_mat[3].imag;

    float t_r = r00 + r11;
    float t_i = i00 + i11;

    float det_r = r00 * r11 - i00 * i11 - (r01 * r10 - i01 * i10);
    float det_i = r00 * i11 + i00 * r11 - (r01 * i10 + i01 * r10);

    float t2_r = t_r * t_r - t_i * t_i;
    float t2_i = 2 * t_r * t_i;

    float disc_r = t2_r - 4.0f * det_r;
    float disc_i = t2_i - 4.0f * det_i;

    float disc_mag = sqrtf(disc_r * disc_r + disc_i * disc_i);
    float disc_ang = atan2f(disc_i, disc_r);

    float sqrt_disc_r = sqrtf(disc_mag) * cosf(disc_ang / 2.0f);
    float sqrt_disc_i = sqrtf(disc_mag) * sinf(disc_ang / 2.0f);

    eigenvalues[0].real = ((t_r + sqrt_disc_r) / 2.0f);
    eigenvalues[0].imag = ((t_i + sqrt_disc_i) / 2.0f);

    eigenvalues[1].real = ((t_r - sqrt_disc_r) / 2.0f);
    eigenvalues[1].imag = ((t_i - sqrt_disc_i) / 2.0f);
}

static void bm_sort_eigenvalues(Complex *eigenvalues, Complex *eigenvectors, int n) {
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

static float bm_vel_clip(float v) {
    if (v < -30.0f) return -30.0f;
    if (v > 30.0f) return 30.0f;
    return v;
}

// --- Benchmark Top Function ---
void esprit_benchmark(const Complex rec_signal[N_SAMPLES], float out_estimates[2]) {
    int N = N_SAMPLES;
    int k = 2;
    int l = N / 2;

    Complex autocorrelation_matrix[100];
    bm_spatial_smoothing(rec_signal, N, autocorrelation_matrix);

    Complex eigenvalues_all[10];
    Complex eigenvectors_all[100];
    bm_qr_algorithm_eig_single(autocorrelation_matrix, l, 10, eigenvalues_all, eigenvectors_all);

    bm_sort_eigenvalues(eigenvalues_all, eigenvectors_all, l);

    int rows_sub = l - 1;
    Complex subA1[18];
    Complex subB1[18];
    for(int j=0; j<k; j++) {
        for(int i=0; i<rows_sub; i++) {
            subA1[i*k+j] = eigenvectors_all[i*l+j];
            subB1[i*k+j] = eigenvectors_all[(i+1)*l+j];
        }
    }

    Complex pinvA1[18];
    bm_svd_pinv_complex_k2(subA1, rows_sub, pinvA1);
    Complex phi_mat[4];

    for(int i=0; i<k; i++) {
        int i_rows = i * rows_sub;
        for(int j=0; j<k; j++) {
            float sum_r = 0.0f, sum_i = 0.0f;
            for(int m=0; m<rows_sub; m++) {
                float pr = pinvA1[i_rows + m].real, pi = pinvA1[i_rows + m].imag;
                float br = subB1[m*k+j].real, bi = subB1[m*k+j].imag;
                sum_r += pr * br - pi * bi;
                sum_i += pr * bi + pi * br;
            }
            phi_mat[i*k+j].real = sum_r;
            phi_mat[i*k+j].imag = sum_i;
        }
    }

    Complex phi_eigs[2];
    bm_eign_cal_ES(phi_mat, phi_eigs);
    float estimates[2];
    for(int i=0; i<k; i++) {
        float angle = atan2f(phi_eigs[i].imag, phi_eigs[i].real);
        estimates[i] = -angle / 0.005f;
    }

    if (estimates[0] > estimates[1]) {
        float temp = estimates[0];
        estimates[0] = estimates[1];
        estimates[1] = temp;
    }

    out_estimates[0] = bm_vel_clip(estimates[0]);
    out_estimates[1] = bm_vel_clip(estimates[1]);
}

int main() {
    int error_count = 0;

    // Variables for RMSE calculation
    float snr_sq_err[3] = {0.0f, 0.0f, 0.0f};
    int snr_processed_count[3] = {0, 0, 0};
    int snr_list[3] = {-20, 0, 20};

    for (int s = 0; s < NUM_SAMPLES; s++) {
        const ESPRIT_Sample *current = &dataset[s];

        // 1. Run Benchmark (Software Model)
        float bm_estimates[2];
        esprit_benchmark(current->data, bm_estimates);

        // 2. Setup Hardware Streams
        hls::stream<axis_data> in_stream, out_stream;
        axis_data local_read, local_write;

        // Send input stream
        for (int i = 0; i < N_SAMPLES; i++) {
            local_read.data = current->data[i].real;
            local_read.keep = -1;
            local_read.last = 0;
            in_stream.write(local_read);

            local_read.data = current->data[i].imag;
            local_read.keep = -1;
            local_read.last = (i == N_SAMPLES - 1) ? 1 : 0;
            in_stream.write(local_read);
        }

        // 3. Call
        esprit_hls(in_stream, out_stream);

        // 4. Read output streams
        local_write = out_stream.read();
        float hw_clipped0 = local_write.data;

        local_write = out_stream.read();
        float hw_clipped1 = local_write.data;

        // 5. Compare outputs
               if (fabs(hw_clipped0 - bm_estimates[0]) > 0.05f || fabs(hw_clipped1 - bm_estimates[1]) > 0.05f) {
                   printf("Error at sample %d (SNR %d): HW=(%f, %f) BM=(%f, %f)\n",
                          current->sample_id, current->snr, hw_clipped0, hw_clipped1, bm_estimates[0], bm_estimates[1]);
                   error_count++;
               }
               else{
               	printf("Hw-Sw matches at %d\n ",current->sample_id);
               }

        printf("Sample %d (SNR %d): HW: %.10f, %.10f  , SW: %.10f, %.10f\n\n", current->sample_id, current->snr, hw_clipped0, hw_clipped1 , bm_estimates[0] , bm_estimates[1]);

        // --- RMSE Accumulation ---
        float gt[2] = {current->gt_vel1, current->gt_vel2};
        if (gt[0] > gt[1]) {
            float temp = gt[0];
            gt[0] = gt[1];
            gt[1] = temp;
        }

        // Identify SNR index
        int snr_idx = -1;
        for(int i=0; i<3; i++) {
            if(current->snr == snr_list[i]) {
                snr_idx = i;
                break;
            }
        }

        if(snr_idx != -1) {
            snr_sq_err[snr_idx] += (hw_clipped0 - gt[0]) * (hw_clipped0 - gt[0]);
            snr_sq_err[snr_idx] += (hw_clipped1 - gt[1]) * (hw_clipped1 - gt[1]);
            snr_processed_count[snr_idx]++;
        }


    }

    if (error_count > 0) {
        printf("\nSimulation Failed with %d errors.\n", error_count);
      //  return 1;
    }

    printf("\n========================================\n");
    printf("   RMSE RESULTS FOR EACH SNR\n");
    printf("========================================\n");
    for(int i=0; i<3; i++) {
        if(snr_processed_count[i] > 0) {
            float rmse = sqrtf(snr_sq_err[i] / (snr_processed_count[i] * 2.0f));
            printf("SNR %4d: RMSE = %.10f\n", snr_list[i], rmse);
        }
    }

    printf("\nNo Error!\n");
    return 0;
}
