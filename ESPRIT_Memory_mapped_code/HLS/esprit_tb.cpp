#include "esprit.h"
#include <math.h>
#include <stdio.h>
#include "inputs.h"

// =====================================================================
// --TESTBENCH-----
// =====================================================================

static inline float bm_vel_clip(float x) {
    if (x > 30.0f) return 30.0f;
    if (x < -30.0f) return -30.0f;
    return x;
}

static inline float bm_fast_atan2f(float y, float x) {
    if (x == 0.0f && y == 0.0f) return 0.0f;
    float ax = fabsf(x);
    float ay = fabsf(y);
    float a = (ax < ay) ? (ax / ay) : (ay / ax);
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (ay > ax) r = 1.570796327f - r;
    if (x < 0.0f) r = 3.141592654f - r;
    if (y < 0.0f) r = -r;
    return r;
}

static inline Complex bm_conj(Complex a) {
    return {a.real, -a.imag};
}

static void bm_spatial_smoothing(const Complex *rec_signal, int N, Complex *output_matrix) {
    int l = N / 2;
    int subarr = N + 1 - l;
    float accum_r[100] = {0.0f}, accum_i[100] = {0.0f};

    for (int i = 0; i < subarr; i++) {
        const Complex *sig_ptr = &rec_signal[i];
        for(int j = 0; j < l; j++) {
            for(int k = 0; k < l; k++) {
                accum_r[j*l + k] += sig_ptr[j].real * sig_ptr[k].real + sig_ptr[j].imag * sig_ptr[k].imag;
                accum_i[j*l + k] += sig_ptr[j].real * sig_ptr[k].imag - sig_ptr[j].imag * sig_ptr[k].real;
            }
        }
    }

    float inv_subarr = 1.0f / (float)subarr;
    for(int k = 0; k < 100; k++) {
        output_matrix[k].real = accum_r[k] * inv_subarr;
        output_matrix[k].imag = accum_i[k] * inv_subarr;
    }
}


static void bm_givens_rotation(Complex a, Complex b, Complex *c, Complex *s) {
    float hypo = sqrtf(a.real*a.real + a.imag*a.imag + b.real*b.real + b.imag*b.imag);
    if (hypo == 0.0f) {
        c->real = 1.0f; c->imag = 0.0f;
        s->real = 0.0f; s->imag = 0.0f;
    } else {
        c->real = a.real / hypo; c->imag = a.imag / hypo;
        s->real = b.real / hypo; s->imag = b.imag / hypo;
    }
}

static void bm_qr_givens_S(Complex R[10][10], Complex Q_out[10][10], int n) {
    Complex Q_temp[10][10];

    // Initialize Q_temp as identity
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++) {
            Q_temp[i][j].real = (i == j) ? 1.0f : 0.0f;
            Q_temp[i][j].imag = 0.0f;
        }

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            Complex cosf, sinf;
            bm_givens_rotation(R[i][i], R[j][i], &cosf, &sinf);

            for (int k = 0; k < n; k++) {
                float Ri_r = R[i][k].real, Ri_i = R[i][k].imag;
                float Rj_r = R[j][k].real, Rj_i = R[j][k].imag;

                // new R[i][k] = conj(c)*Ri + conj(s)*Rj
                R[i][k].real =  Ri_r*cosf.real + Ri_i*cosf.imag + Rj_r*sinf.real + Rj_i*sinf.imag;
                R[i][k].imag = -Ri_r*cosf.imag + Ri_i*cosf.real - Rj_r*sinf.imag + Rj_i*sinf.real;
                // new R[j][k] = -s*Ri + c*Rj
                R[j][k].real = -Ri_r*sinf.real + Ri_i*sinf.imag + Rj_r*cosf.real - Rj_i*cosf.imag;
                R[j][k].imag = -Ri_r*sinf.imag - Ri_i*sinf.real + Rj_r*cosf.imag + Rj_i*cosf.real;

                float Qi_r = Q_temp[i][k].real, Qi_i = Q_temp[i][k].imag;
                float Qj_r = Q_temp[j][k].real, Qj_i = Q_temp[j][k].imag;

                // new Q_temp[i][k] = conj(c)*Qi + conj(s)*Qj
                Q_temp[i][k].real =  Qi_r*cosf.real + Qi_i*cosf.imag + Qj_r*sinf.real + Qj_i*sinf.imag;
                Q_temp[i][k].imag = -Qi_r*cosf.imag + Qi_i*cosf.real - Qj_r*sinf.imag + Qj_i*sinf.real;
                // new Q_temp[j][k] = -s*Qi + c*Qj
                Q_temp[j][k].real = -Qi_r*sinf.real + Qi_i*sinf.imag + Qj_r*cosf.real - Qj_i*cosf.imag;
                Q_temp[j][k].imag = -Qi_r*sinf.imag - Qi_i*sinf.real + Qj_r*cosf.imag + Qj_i*cosf.real;
            }
        }
    }

    // Q_out = conj_transpose(Q_temp)  (mirrors transpose_conj_matrix_2d)
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++) {
            Q_out[i][j].real =  Q_temp[j][i].real;
            Q_out[i][j].imag = -Q_temp[j][i].imag;
        }
}

static void bm_qr_algorithm_eig_single(const Complex *A_input, int n, int max_iterations,
                                        Complex *eigenvalues, Complex *eigenvectors) {
    Complex A[10][10], Q[10][10], R[10][10], TempA[10][10], TempQTotal[10][10];

    // Copy 1D input to 2D local
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            A[i][j] = A_input[i*n + j];

    // Initialize TempQTotal as identity
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++) {
            TempQTotal[i][j].real = (i == j) ? 1.0f : 0.0f;
            TempQTotal[i][j].imag = 0.0f;
        }

    for (int iter = 0; iter < max_iterations; iter++) {
        // R = A (copy), then QR-decompose: R becomes upper-triangular, Q returned
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                R[i][j] = A[i][j];

        bm_qr_givens_S(R, Q, n);  // R is modified in-place; Q is the orthogonal factor

        // A = R * Q
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++) {
                float sr = 0.0f, si = 0.0f;
                for(int k = 0; k < n; k++) {
                    sr += R[i][k].real * Q[k][j].real - R[i][k].imag * Q[k][j].imag;
                    si += R[i][k].real * Q[k][j].imag + R[i][k].imag * Q[k][j].real;
                }
                A[i][j] = {sr, si};
            }

        // TempA = TempQTotal,  TempQTotal = TempA * Q
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                TempA[i][j] = TempQTotal[i][j];

        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++) {
                float sr = 0.0f, si = 0.0f;
                for(int k = 0; k < n; k++) {
                    sr += TempA[i][k].real * Q[k][j].real - TempA[i][k].imag * Q[k][j].imag;
                    si += TempA[i][k].real * Q[k][j].imag + TempA[i][k].imag * Q[k][j].real;
                }
                TempQTotal[i][j] = {sr, si};
            }
    }

    // Output eigenvalues (diagonal) and eigenvectors
    for(int i = 0; i < n; i++) {
        eigenvalues[i] = A[i][i];
        for(int j = 0; j < n; j++)
            eigenvectors[i*n + j] = TempQTotal[i][j];
    }
}

static void bm_sort_eigenvalues(Complex *eigenvalues, Complex *eigenvectors, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            float m1 = eigenvalues[j].real*eigenvalues[j].real     + eigenvalues[j].imag*eigenvalues[j].imag;
            float m2 = eigenvalues[j+1].real*eigenvalues[j+1].real + eigenvalues[j+1].imag*eigenvalues[j+1].imag;
            if (m1 < m2) {
                Complex tV = eigenvalues[j]; eigenvalues[j] = eigenvalues[j+1]; eigenvalues[j+1] = tV;
                for (int r = 0; r < n; r++) {
                    Complex tP = eigenvectors[r*n+j];
                    eigenvectors[r*n+j]   = eigenvectors[r*n+j+1];
                    eigenvectors[r*n+j+1] = tP;
                }
            }
        }
    }
}

static void bm_svd_pinv_complex_k2(Complex *A, int m, Complex *pinvA) {
    float H[2][2][2] = {0};
    for(int k = 0; k < m; ++k) {
        for(int i = 0; i < 2; ++i) {
            float a_ik_r =  A[k*2+i].real;
            float a_ik_i = -A[k*2+i].imag;
            for(int j = 0; j < 2; ++j) {
                float a_kj_r = A[k*2+j].real, a_kj_i = A[k*2+j].imag;
                H[i][j][0] += (a_ik_r * a_kj_r - a_ik_i * a_kj_i);
                H[i][j][1] += (a_ik_r * a_kj_i + a_ik_i * a_kj_r);
            }
        }
    }

    float det_r = H[0][0][0]*H[1][1][0] - H[0][0][1]*H[1][1][1]
                - (H[0][1][0]*H[1][0][0] - H[0][1][1]*H[1][0][1]);
    float det_i = H[0][0][0]*H[1][1][1] + H[0][0][1]*H[1][1][0]
                - (H[0][1][0]*H[1][0][1] + H[0][1][1]*H[1][0][0]);
    float det_mag_sq = det_r*det_r + det_i*det_i;
    float inv_det_r =  det_r / det_mag_sq;
    float inv_det_i = -det_i / det_mag_sq;

    float H_inv[2][2][2];
    H_inv[0][0][0] =  (H[1][1][0]*inv_det_r - H[1][1][1]*inv_det_i);
    H_inv[0][0][1] =  (H[1][1][0]*inv_det_i + H[1][1][1]*inv_det_r);
    H_inv[1][1][0] =  (H[0][0][0]*inv_det_r - H[0][0][1]*inv_det_i);
    H_inv[1][1][1] =  (H[0][0][0]*inv_det_i + H[0][0][1]*inv_det_r);
    H_inv[0][1][0] = -(H[0][1][0]*inv_det_r - H[0][1][1]*inv_det_i);
    H_inv[0][1][1] = -(H[0][1][0]*inv_det_i + H[0][1][1]*inv_det_r);
    H_inv[1][0][0] = -(H[1][0][0]*inv_det_r - H[1][0][1]*inv_det_i);
    H_inv[1][0][1] = -(H[1][0][0]*inv_det_i + H[1][0][1]*inv_det_r);

    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < m; ++j) {
            float sum_r = 0, sum_i = 0;
            for(int k = 0; k < 2; ++k) {
                float ah_kj_r =  A[j*2+k].real;
                float ah_kj_i = -A[j*2+k].imag;
                sum_r += H_inv[i][k][0] * ah_kj_r - H_inv[i][k][1] * ah_kj_i;
                sum_i += H_inv[i][k][0] * ah_kj_i + H_inv[i][k][1] * ah_kj_r;
            }
            pinvA[i*m+j].real = sum_r;
            pinvA[i*m+j].imag = sum_i;
        }
    }
}

static void bm_eign_cal_ES(Complex *phi_mat, Complex *eigenvalues) {
    float t_r = phi_mat[0].real + phi_mat[3].real;
    float t_i = phi_mat[0].imag + phi_mat[3].imag;
    float det_r = phi_mat[0].real*phi_mat[3].real - phi_mat[0].imag*phi_mat[3].imag
                - (phi_mat[1].real*phi_mat[2].real - phi_mat[1].imag*phi_mat[2].imag);
    float det_i = phi_mat[0].real*phi_mat[3].imag + phi_mat[0].imag*phi_mat[3].real
                - (phi_mat[1].real*phi_mat[2].imag + phi_mat[1].imag*phi_mat[2].real);

    float disc_r = t_r*t_r - t_i*t_i - 4.0f*det_r;
    float disc_i = 2.0f*t_r*t_i - 4.0f*det_i;
    float mag = sqrtf(sqrtf(disc_r*disc_r + disc_i*disc_i));
    float ang = atan2f(disc_i, disc_r) * 0.5f;
    float sdr = mag * cosf(ang), sdi = mag * sinf(ang);

    eigenvalues[0].real = (t_r + sdr) * 0.5f;
    eigenvalues[0].imag = (t_i + sdi) * 0.5f;
    eigenvalues[1].real = (t_r - sdr) * 0.5f;
    eigenvalues[1].imag = (t_i - sdi) * 0.5f;
}

// =====================================================================
// ---  Software run---
// =====================================================================

void run_sw_benchmark(const Complex *input, float *sw_output) {
    int l = N_SAMPLES / 2;  // l = 10, matches HW
    int k = 2;
    Complex auto_mat[100], e_vals[10], e_vecs[100];

    bm_spatial_smoothing(input, N_SAMPLES, auto_mat);
    bm_qr_algorithm_eig_single(auto_mat, l, 10, e_vals, e_vecs);
    bm_sort_eigenvalues(e_vals, e_vecs, l);

    int rows_sub = l - 1;  // 9
    Complex subA[18], subB[18], pinvA[18], phi[4], phi_eigs[2];

    for(int j = 0; j < k; j++) {
        for(int i = 0; i < rows_sub; i++) {
            subA[i*k+j] = e_vecs[i*l+j];
            subB[i*k+j] = e_vecs[(i+1)*l+j];
        }
    }

    bm_svd_pinv_complex_k2(subA, rows_sub, pinvA);

    for(int i = 0; i < k; i++) {
        for(int j = 0; j < k; j++) {
            float sr = 0, si = 0;
            for(int m = 0; m < rows_sub; m++) {
                sr += pinvA[i*rows_sub+m].real * subB[m*k+j].real
                    - pinvA[i*rows_sub+m].imag * subB[m*k+j].imag;
                si += pinvA[i*rows_sub+m].real * subB[m*k+j].imag
                    + pinvA[i*rows_sub+m].imag * subB[m*k+j].real;
            }
            phi[i*k+j].real = sr;
            phi[i*k+j].imag = si;
        }
    }

    bm_eign_cal_ES(phi, phi_eigs);

    // FIX 2: apply same vel_clip and sort as HW
    sw_output[0] = bm_vel_clip(-bm_fast_atan2f(phi_eigs[0].imag, phi_eigs[0].real) / 0.005f);
    sw_output[1] = bm_vel_clip(-bm_fast_atan2f(phi_eigs[1].imag, phi_eigs[1].real) / 0.005f);
    if (sw_output[0] > sw_output[1]) { float t = sw_output[0]; sw_output[0] = sw_output[1]; sw_output[1] = t; }
}

// =====================================================================
// --- MAIN TESTBENCH ---
// =====================================================================
int main() {

    float hw_out[2], sw_out[2];   // FIX 3: single declaration (removed duplicate inside loop)
    int error_count = 0;

    const int snr_levels[] = {-20, 0, 20};
    double total_sq_err_snr[3] = {0, 0, 0};
    int count_snr[3] = {0, 0, 0};

    printf("========================================================================================================================\n");
    printf("%-7s | %-4s | %-20s | %-20s | %-20s\n", "Sample", "SNR", "HW (Hardware)", "SW (Benchmark)", "GT (Ground Truth)");
    printf("------------------------------------------------------------------------------------------------------------------------\n");

    for (int s = 0; s < NUM_SAMPLES; s++) {

        const ESPRIT_Sample *cur = &dataset[s];

        Complex input[N_SAMPLES];
        for(int i = 0; i < N_SAMPLES; i++) input[i] = cur->data[i];

        // 1. Run Hardware
        esprit_hls(input, hw_out);
        if (hw_out[0] > hw_out[1]) { float t = hw_out[0]; hw_out[0] = hw_out[1]; hw_out[1] = t; }

        // 2. Run Software Reference
        run_sw_benchmark(cur->data, sw_out);

        // 3. Ground Truth
        float gt[2] = {cur->gt_vel1, cur->gt_vel2};
        if (gt[0] > gt[1]) { float t = gt[0]; gt[0] = gt[1]; gt[1] = t; }

        // 4. Check HW vs SW
        if (fabs(hw_out[0] - sw_out[0]) > 0.01f || fabs(hw_out[1] - sw_out[1]) > 0.01f) {
            printf(">>> ERROR: HW/SW Mismatch at Sample %d! HW={%.4f, %.4f}, SW={%.4f, %.4f}\n",
                   s, hw_out[0], hw_out[1], sw_out[0], sw_out[1]);
            error_count++;
        } else {
            printf("Sample %d: hw-sw values match\n", s);
        }

        // 5. Print detailed row
        printf("S-%-5d | %-4d | {%8.4f, %8.4f} | {%8.4f, %8.4f} | {%8.4f, %8.4f}\n\n",
               s, cur->snr, hw_out[0], hw_out[1], sw_out[0], sw_out[1], gt[0], gt[1]);

        // 6. Accumulate Squared Errors by SNR
        int snr_idx = -1;
        if      (cur->snr == -20) snr_idx = 0;
        else if (cur->snr ==   0) snr_idx = 1;
        else if (cur->snr ==  20) snr_idx = 2;

        if (snr_idx != -1) {
            float err0 = hw_out[0] - gt[0];
            float err1 = hw_out[1] - gt[1];
            total_sq_err_snr[snr_idx] += (double)(err0*err0 + err1*err1);
            count_snr[snr_idx]++;
        }
    }

    // --- Final RMSE Summary ---
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    printf("\nRMSE Summary Table:\n");
    printf("----------------------------------------------------------\n");
    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            double mse  = total_sq_err_snr[i] / (count_snr[i] * 2.0);
            double rmse = sqrt(mse);
            printf("RMSE velocity for esprit at SNR %-4d dB: %f\n", snr_levels[i], (float)rmse);
        }
    }
    printf("----------------------------------------------------------\n");

    if (error_count == 0)
        printf("\nAll %d samples: HW matches SW. PASS\n", NUM_SAMPLES);
    else
        printf("\n%d sample(s) failed HW/SW match. FAIL\n", error_count);

    return (error_count == 0) ? 0 : 1;
}
