#include "esprit.h"

// Fast math approximation for HLS efficiency
static inline float fast_atan2f(float y, float x) {
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

static inline Complex conj_complex(Complex a) {
    Complex result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

void matmul_complex_2d(const Complex A[10][10], const Complex B[10][10], Complex C[10][10], int n) {
    #pragma HLS INLINE
    for(int i=0; i<10; i++) {
        #pragma HLS UNROLL
        for(int j=0; j<10; j++) {
            C[i][j].real = 0.0f;
            C[i][j].imag = 0.0f;
        }
    }

    for (int i = 0; i < 10; i++) {
        #pragma HLS UNROLL
        for (int k = 0; k < 10; k++) {
            #pragma HLS PIPELINE II=1
            float ar = A[i][k].real, ai = A[i][k].imag;
            for (int j = 0; j < 10; j++) {
                float br = B[k][j].real, bi = B[k][j].imag;
                C[i][j].real += (ar * br - ai * bi);
                C[i][j].imag += (ar * bi + ai * br);
            }
        }
    }
}

void transpose_conj_matrix_2d(const Complex A[10][10], Complex B[10][10], int n) {
    #pragma HLS INLINE
    for(int i=0; i<10; i++) {
        #pragma HLS UNROLL
        for(int j=0; j<10; j++) {
            B[j][i] = conj_complex(A[i][j]);
        }
    }
}

void copy_matrix_2d(const Complex src[10][10], Complex dst[10][10], int n) {
    #pragma HLS INLINE
    for(int i=0; i<10; i++) {
        #pragma HLS UNROLL
        for(int j=0; j<10; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

void identity_matrix_2d(Complex A[10][10], int n) {
    #pragma HLS INLINE
    for(int i=0; i<10; i++) {
        #pragma HLS UNROLL
        for(int j=0; j<10; j++) {
            A[i][j].real = (i == j) ? 1.0f : 0.0f;
            A[i][j].imag = 0.0f;
        }
    }
}

void spatial_smoothing(const Complex *rec_signal, int N, Complex *output_matrix) {
    #pragma HLS INLINE off
    int l = N / 2;
    int subarr = N + 1 - l;
    float accum_r[100], accum_i[100];

    for(int i=0; i<100; i++) {
        #pragma HLS PIPELINE II=1
        accum_r[i] = 0.0f; accum_i[i] = 0.0f;
    }

    for (int i = 0; i < subarr; i++) {
        const Complex *sig_ptr = &rec_signal[i];
        for(int j=0; j<l; j++) {
            #pragma HLS PIPELINE II=1
            float r_j_r = sig_ptr[j].real;
            float r_j_i = sig_ptr[j].imag;
            for(int k=0; k<l; k++) {
                float r_k_r = sig_ptr[k].real;
                float r_k_i = sig_ptr[k].imag;
                accum_r[j*l + k] += r_j_r * r_k_r + r_j_i * r_k_i;
                accum_i[j*l + k] += r_j_r * r_k_i - r_j_i * r_k_r;
            }
        }
    }

    float inv_subarr = 1.0f / (float)subarr;
    for(int k=0; k<100; k++) {
        #pragma HLS PIPELINE II=1
        output_matrix[k].real = accum_r[k] * inv_subarr;
        output_matrix[k].imag = accum_i[k] * inv_subarr;
    }
}

void givensrotation_new(Complex a, Complex b, Complex *cos_out, Complex *sin_out) {
    #pragma HLS INLINE off
    float hypo = sqrtf(a.real*a.real + a.imag*a.imag + b.real*b.real + b.imag*b.imag);
    if (hypo == 0.0f) {
        cos_out->real = 1.0f; cos_out->imag = 0.0f;
        sin_out->real = 0.0f; sin_out->imag = 0.0f;
    } else {
        cos_out->real = a.real / hypo; cos_out->imag = a.imag / hypo;
        sin_out->real = b.real / hypo; sin_out->imag = b.imag / hypo;
    }
}

void qr_givens_S_2d(Complex R_out[10][10], Complex Q_out[10][10], int n) {
    #pragma HLS INLINE
    Complex Q_temp[10][10];
    identity_matrix_2d(Q_temp, n);

    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 10; j++) {
            Complex cosf, sinf;
            givensrotation_new(R_out[i][i], R_out[j][i], &cosf, &sinf);

            for (int k = 0; k < 10; k++) {
                #pragma HLS UNROLL
                float Ri_r = R_out[i][k].real, Ri_i = R_out[i][k].imag;
                float Rj_r = R_out[j][k].real, Rj_i = R_out[j][k].imag;

                R_out[i][k].real = Ri_r*cosf.real - Ri_i*(-cosf.imag) + Rj_r*sinf.real - Rj_i*(-sinf.imag);
                R_out[i][k].imag = Ri_r*(-cosf.imag) + Ri_i*cosf.real + Rj_r*(-sinf.imag) + Rj_i*sinf.real;
                R_out[j][k].real = Ri_r*(-sinf.real) - Ri_i*(-sinf.imag) + Rj_r*cosf.real - Rj_i*cosf.imag;
                R_out[j][k].imag = Ri_r*(-sinf.imag) + Ri_i*(-sinf.real) + Rj_r*cosf.imag + Rj_i*cosf.real;

                float Qi_r = Q_temp[i][k].real, Qi_i = Q_temp[i][k].imag;
                float Qj_r = Q_temp[j][k].real, Qj_i = Q_temp[j][k].imag;

                Q_temp[i][k].real = Qi_r*cosf.real - Qi_i*(-cosf.imag) + Qj_r*sinf.real - Qj_i*(-sinf.imag);
                Q_temp[i][k].imag = Qi_r*(-cosf.imag) + Qi_i*cosf.real + Qj_r*(-sinf.imag) + Qj_i*sinf.real;
                Q_temp[j][k].real = Qi_r*(-sinf.real) - Qi_i*(-sinf.imag) + Qj_r*cosf.real - Qj_i*cosf.imag;
                Q_temp[j][k].imag = Qi_r*(-sinf.imag) + Qi_i*(-sinf.real) + Qj_r*cosf.imag + Qj_i*cosf.real;
            }
        }
    }
    transpose_conj_matrix_2d(Q_temp, Q_out, n);
}

void qr_algorithm_eig_single(const Complex *A_input, int n, int max_iterations, Complex *eigenvalues, Complex *eigenvectors) {
    #pragma HLS INLINE off
    Complex A[10][10], Q[10][10], R[10][10], TempA[10][10], TempQTotal[10][10];


#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=Q complete dim=2
#pragma HLS ARRAY_PARTITION variable=R complete dim=2
#pragma HLS ARRAY_PARTITION variable=TempA complete dim=2
#pragma HLS ARRAY_PARTITION variable=TempQTotal complete dim=2



    // Initial copy from 1D input to 2D local
    for(int i=0; i<10; i++) {
        for(int j=0; j<10; j++) {
            #pragma HLS PIPELINE II=1
            A[i][j] = A_input[i*10 + j];
        }
    }

    // Initialize  as identity
    identity_matrix_2d(TempQTotal, 10);

    for (int iter = 0; iter < 10; iter++) {
        copy_matrix_2d(A, R, 10);
        qr_givens_S_2d(R, Q, 10);
        matmul_complex_2d(R, Q, A, 10);
        copy_matrix_2d(TempQTotal, TempA, 10);
        matmul_complex_2d(TempA, Q, TempQTotal, 10);
    }

    // Output results back to 1D
    for(int i=0; i<10; i++) {
	#pragma HLS PIPELINE II=1
        eigenvalues[i] = A[i][i];
        for(int j=0; j<10; j++) eigenvectors[i*10+j] = TempQTotal[i][j];
    }
}

void svd_pinv_complex_k2(Complex *A, int m, Complex *pinvA) {
    #pragma HLS INLINE off
    float H[2][2][2] = {0};
    for(int k=0; k<m; ++k) {
        for(int i=0; i<2; ++i) {
            float a_ik_r = A[k*2+i].real, a_ik_i = -A[k*2+i].imag;
            for(int j=0; j<2; ++j) {
                #pragma HLS PIPELINE II=1
                float a_kj_r = A[k*2+j].real, a_kj_i = A[k*2+j].imag;
                H[i][j][0] += (a_ik_r * a_kj_r - a_ik_i * a_kj_i);
                H[i][j][1] += (a_ik_r * a_kj_i + a_ik_i * a_kj_r);
            }
        }
    }

    float det_r = H[0][0][0]*H[1][1][0] - H[0][0][1]*H[1][1][1] - (H[0][1][0]*H[1][0][0] - H[0][1][1]*H[1][0][1]);
    float det_i = H[0][0][0]*H[1][1][1] + H[0][0][1]*H[1][1][0] - (H[0][1][0]*H[1][0][1] + H[0][1][1]*H[1][0][0]);
    float det_mag_sq = det_r*det_r + det_i*det_i;
    float inv_det_r = det_r / det_mag_sq;
    float inv_det_i = -det_i / det_mag_sq;

    float H_inv[2][2][2];
    H_inv[0][0][0] = (H[1][1][0]*inv_det_r - H[1][1][1]*inv_det_i);
    H_inv[0][0][1] = (H[1][1][0]*inv_det_i + H[1][1][1]*inv_det_r);
    H_inv[1][1][0] = (H[0][0][0]*inv_det_r - H[0][0][1]*inv_det_i);
    H_inv[1][1][1] = (H[0][0][0]*inv_det_i + H[0][0][1]*inv_det_r);
    H_inv[0][1][0] = -(H[0][1][0]*inv_det_r - H[0][1][1]*inv_det_i);
    H_inv[0][1][1] = -(H[0][1][0]*inv_det_i + H[0][1][1]*inv_det_r);
    H_inv[1][0][0] = -(H[1][0][0]*inv_det_r - H[1][0][1]*inv_det_i);
    H_inv[1][0][1] = -(H[1][0][0]*inv_det_i + H[1][0][1]*inv_det_r);

    for(int i=0; i<2; ++i) {
        for(int j=0; j<m; ++j) {
            #pragma HLS PIPELINE II=1
            float sum_r = 0, sum_i = 0;
            for(int k=0; k<2; ++k) {
                float ah_kj_r = A[j*2+k].real, ah_kj_i = -A[j*2+k].imag;
                sum_r += H_inv[i][k][0] * ah_kj_r - H_inv[i][k][1] * ah_kj_i;
                sum_i += H_inv[i][k][0] * ah_kj_i + H_inv[i][k][1] * ah_kj_r;
            }
            pinvA[i*m+j].real = sum_r; pinvA[i*m+j].imag = sum_i;
        }
    }
}

void eign_cal_ES(Complex *phi_mat, Complex *eigenvalues) {
    #pragma HLS INLINE off
    float t_r = phi_mat[0].real + phi_mat[3].real;
    float t_i = phi_mat[0].imag + phi_mat[3].imag;
    float det_r = phi_mat[0].real*phi_mat[3].real - phi_mat[0].imag*phi_mat[3].imag - (phi_mat[1].real*phi_mat[2].real - phi_mat[1].imag*phi_mat[2].imag);
    float det_i = phi_mat[0].real*phi_mat[3].imag + phi_mat[0].imag*phi_mat[3].real - (phi_mat[1].real*phi_mat[2].imag + phi_mat[1].imag*phi_mat[2].real);

    float disc_r = t_r*t_r - t_i*t_i - 4.0f*det_r;
    float disc_i = 2.0f*t_r*t_i - 4.0f*det_i;
    float mag = sqrtf(sqrtf(disc_r*disc_r + disc_i*disc_i));
    float ang = atan2f(disc_i, disc_r) * 0.5f;
    float sdr = mag * cosf(ang), sdi = mag * sinf(ang);

    eigenvalues[0].real = (t_r + sdr)*0.5f; eigenvalues[0].imag = (t_i + sdi)*0.5f;
    eigenvalues[1].real = (t_r - sdr)*0.5f; eigenvalues[1].imag = (t_i - sdi)*0.5f;
}

void sort_eigenvalues(Complex *eigenvalues, Complex *eigenvectors, int n) {
    #pragma HLS INLINE off
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            float m1 = eigenvalues[j].real*eigenvalues[j].real + eigenvalues[j].imag*eigenvalues[j].imag;
            float m2 = eigenvalues[j+1].real*eigenvalues[j+1].real + eigenvalues[j+1].imag*eigenvalues[j+1].imag;
            if (m1 < m2) {
                Complex tV = eigenvalues[j]; eigenvalues[j] = eigenvalues[j+1]; eigenvalues[j+1] = tV;
                for (int r = 0; r < n; r++) {
                    #pragma HLS PIPELINE II=1
                    Complex tP = eigenvectors[r*n+j]; eigenvectors[r*n+j] = eigenvectors[r*n+j+1]; eigenvectors[r*n+j+1] = tP;
                }
            }
        }
    }
}

float vel_clip(float v) {
    if (v < -30.0f) return -30.0f;
    if (v > 30.0f) return 30.0f;
    return v;
}

void esprit_hls(Complex *in_mem, float *out_mem) {
    // --- MEMORY MAPPED INTERFACES ---
    // Control interface (Slave AXI-Lite)
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE m_axi port=in_mem  offset=slave depth=20
    #pragma HLS INTERFACE m_axi port=out_mem offset=slave depth=2


    Complex rec_signal[N_SAMPLES];
    #pragma HLS ARRAY_PARTITION variable=rec_signal type=complete dim=0
    memcpy(rec_signal, (const Complex*)in_mem, N_SAMPLES * sizeof(Complex));

    int l = N_SAMPLES / 2, k = 2;
    Complex auto_mat[100], e_vals[10], e_vecs[100];

    spatial_smoothing(rec_signal, N_SAMPLES, auto_mat);
    qr_algorithm_eig_single(auto_mat, l, 10, e_vals, e_vecs);
    sort_eigenvalues(e_vals, e_vecs, l);

    int rows_sub = l - 1;
    Complex subA[18], subB[18], pinvA[18], phi[4], phi_eigs[2];

    for(int j=0; j<k; j++) {
        for(int i=0; i<rows_sub; i++) {
            #pragma HLS PIPELINE II=1
            subA[i*k+j] = e_vecs[i*l+j];
            subB[i*k+j] = e_vecs[(i+1)*l+j];
        }
    }

    svd_pinv_complex_k2(subA, rows_sub, pinvA);

    for(int i=0; i<k; i++) {
        for(int j=0; j<k; j++) {
            #pragma HLS PIPELINE II=1
            float sr = 0, si = 0;
            for(int m=0; m<rows_sub; m++) {
                sr += pinvA[i*rows_sub+m].real * subB[m*k+j].real - pinvA[i*rows_sub+m].imag * subB[m*k+j].imag;
                si += pinvA[i*rows_sub+m].real * subB[m*k+j].imag + pinvA[i*rows_sub+m].imag * subB[m*k+j].real;
            }
            phi[i*k+j].real = sr; phi[i*k+j].imag = si;
        }
    }

    eign_cal_ES(phi, phi_eigs);

    float res[2];
    res[0] = vel_clip(-fast_atan2f(phi_eigs[0].imag, phi_eigs[0].real) / 0.005f);
    res[1] = vel_clip(-fast_atan2f(phi_eigs[1].imag, phi_eigs[1].real) / 0.005f);

    if (res[0] > res[1]) { float t = res[0]; res[0] = res[1]; res[1] = t; }

    // 2. Burst Write Results back to DDR
    out_mem[0] = res[0];
    out_mem[1] = res[1];
}
