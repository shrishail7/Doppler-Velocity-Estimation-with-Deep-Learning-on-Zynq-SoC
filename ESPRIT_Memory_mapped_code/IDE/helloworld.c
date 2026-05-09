#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xtime_l.h"
#include "xesprit_hls.h"
#include "xesprit_hls_hw.h"
#include <math.h>

typedef struct {
    float real;
    float imag;
} Complex;

#include "inputs.h"

static const char *snr_label_str[3] = {"-20dB", "0dB", "20dB"};

/* ========================================================== */
/*                  PS (Software) Implementation              */
/* ========================================================== */

static inline Complex ps_conj_complex(Complex a) {
    Complex r; r.real = a.real; r.imag = -a.imag; return r;
}

static void ps_matmul_complex(const Complex *A, const Complex *B, Complex *C, int n) {
    for (int i = 0; i < n; i++) {
        int i_n = i * n;
        for (int j = 0; j < n; j++) {
            float sum_r = 0.0f, sum_i = 0.0f;
            for (int k = 0; k < n; k++) {
                int k_n = k * n;
                float ar = A[i_n+k].real, ai = A[i_n+k].imag;
                float br = B[k_n+j].real, bi = B[k_n+j].imag;
                sum_r += ar*br - ai*bi;
                sum_i += ar*bi + ai*br;
            }
            C[i_n+j].real = sum_r; C[i_n+j].imag = sum_i;
        }
    }
}

static void ps_transpose_conj_matrix(Complex *A, Complex *B, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[j*n+i] = ps_conj_complex(A[i*n+j]);
}

static void ps_copy_matrix(Complex *src, Complex *dst, int n) {
    for (int i = 0; i < n*n; i++) dst[i] = src[i];
}

static void ps_identity_matrix(Complex *A, int n) {
    for (int i = 0; i < n*n; i++) { A[i].real = 0.0f; A[i].imag = 0.0f; }
    for (int i = 0; i < n; i++)   A[i*n+i].real = 1.0f;
}

static void ps_spatial_smoothing(const Complex *rec_signal, int N, Complex *output_matrix) {
    int l      = N / 2;
    int subarr = N + 1 - l;
    float accum_r[100] = {0}, accum_i[100] = {0};

    for (int i = 0; i < subarr; i++) {
        const Complex *sig_ptr = &rec_signal[i];
        for (int j = 0; j < l; j++) {
            int j_l = j * l;
            float r_j_r = sig_ptr[j].real, r_j_i = sig_ptr[j].imag;
            for (int k = 0; k < l; k++) {
                float r_k_r = sig_ptr[k].real, r_k_i = sig_ptr[k].imag;
                accum_r[j_l+k] += r_j_r*r_k_r + r_j_i*r_k_i;
                accum_i[j_l+k] += r_j_r*r_k_i - r_j_i*r_k_r;
            }
        }
    }
    float inv = 1.0f / subarr;
    for (int k = 0; k < l*l; k++) {
        output_matrix[k].real = accum_r[k] * inv;
        output_matrix[k].imag = accum_i[k] * inv;
    }
}

static void ps_givensrotation(Complex a, Complex b, Complex *c_out, Complex *s_out) {
    float hypo = sqrtf(a.real*a.real + a.imag*a.imag + b.real*b.real + b.imag*b.imag);
    if (hypo == 0.0f) {
        c_out->real = 1.0f; c_out->imag = 0.0f;
        s_out->real = 0.0f; s_out->imag = 0.0f;
    } else {
        c_out->real = a.real/hypo; c_out->imag = a.imag/hypo;
        s_out->real = b.real/hypo; s_out->imag = b.imag/hypo;
    }
}

static void ps_qr_givens(const Complex *A, int n, Complex *Q_out, Complex *R_out) {
    ps_copy_matrix((Complex *)A, R_out, n);
    Complex Q_temp[100];
    ps_identity_matrix(Q_temp, n);

    for (int i = 0; i < n-1; i++) {
        int i_n = i*n;
        for (int j = i+1; j < n; j++) {
            int j_n = j*n;
            Complex cosf, sinf;
            ps_givensrotation(R_out[i_n+i], R_out[j_n+i], &cosf, &sinf);

            float c_r=cosf.real, c_i=cosf.imag;
            float s_r=sinf.real, s_i=sinf.imag;
            float cc_r=c_r, cc_i=-c_i;
            float cs_r=s_r, cs_i=-s_i;
            float ns_r=-s_r, ns_i=-s_i;

            for (int k = 0; k < n; k++) {
                float Ri_r=R_out[i_n+k].real, Ri_i=R_out[i_n+k].imag;
                float Rj_r=R_out[j_n+k].real, Rj_i=R_out[j_n+k].imag;

                R_out[i_n+k].real = Ri_r*cc_r - Ri_i*cc_i + Rj_r*cs_r - Rj_i*cs_i;
                R_out[i_n+k].imag = Ri_r*cc_i + Ri_i*cc_r + Rj_r*cs_i + Rj_i*cs_r;
                R_out[j_n+k].real = Ri_r*ns_r - Ri_i*ns_i + Rj_r*c_r  - Rj_i*c_i;
                R_out[j_n+k].imag = Ri_r*ns_i + Ri_i*ns_r + Rj_r*c_i  + Rj_i*c_r;

                float Qi_r=Q_temp[i_n+k].real, Qi_i=Q_temp[i_n+k].imag;
                float Qj_r=Q_temp[j_n+k].real, Qj_i=Q_temp[j_n+k].imag;

                Q_temp[i_n+k].real = Qi_r*cc_r - Qi_i*cc_i + Qj_r*cs_r - Qj_i*cs_i;
                Q_temp[i_n+k].imag = Qi_r*cc_i + Qi_i*cc_r + Qj_r*cs_i + Qj_i*cs_r;
                Q_temp[j_n+k].real = Qi_r*ns_r - Qi_i*ns_i + Qj_r*c_r  - Qj_i*c_i;
                Q_temp[j_n+k].imag = Qi_r*ns_i + Qi_i*ns_r + Qj_r*c_i  + Qj_i*c_r;
            }
        }
    }
    ps_transpose_conj_matrix(Q_temp, Q_out, n);
}

static void ps_qr_algorithm_eig(const Complex *A_input, int n, int max_iter,
                                 Complex *eigenvalues, Complex *eigenvectors) {
    Complex A[100], Q[100], R[100], TmpA[100], TmpQ[100];
    ps_copy_matrix((Complex *)A_input, A, n);
    ps_identity_matrix(eigenvectors, n);

    for (int it = 0; it < max_iter; it++) {
        ps_qr_givens(A, n, Q, R);
        ps_matmul_complex(R, Q, TmpA, n);
        ps_copy_matrix(TmpA, A, n);
        ps_matmul_complex(eigenvectors, Q, TmpQ, n);
        ps_copy_matrix(TmpQ, eigenvectors, n);
    }
    for (int i = 0; i < n; i++) eigenvalues[i] = A[i*n+i];
}

static void ps_svd_pinv_k2(Complex *A, int m, Complex *pinvA) {
    float H[2][2][2];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float sr=0, si=0;
            for (int k = 0; k < m; k++) {
                float ar=A[k*2+i].real, ai=-A[k*2+i].imag;
                float br=A[k*2+j].real, bi= A[k*2+j].imag;
                sr += ar*br - ai*bi; si += ar*bi + ai*br;
            }
            H[i][j][0]=sr; H[i][j][1]=si;
        }
    }
    float tr_r=H[0][0][0]+H[1][1][0], tr_i=H[0][0][1]+H[1][1][1];
    float ad_r=H[0][0][0]*H[1][1][0]-H[0][0][1]*H[1][1][1];
    float ad_i=H[0][0][0]*H[1][1][1]+H[0][0][1]*H[1][1][0];
    float bc_r=H[0][1][0]*H[1][0][0]-H[0][1][1]*H[1][0][1];
    float bc_i=H[0][1][0]*H[1][0][1]+H[0][1][1]*H[1][0][0];
    float det_r=ad_r-bc_r, det_i=ad_i-bc_i;
    float ts_r=tr_r*tr_r-tr_i*tr_i, ts_i=2*tr_r*tr_i;
    float disc_r=ts_r-4*det_r, disc_i=ts_i-4*det_i;
    float dm=sqrtf(disc_r*disc_r+disc_i*disc_i);
    float da=atan2f(disc_i,disc_r);
    float sd_r=sqrtf(dm)*cosf(da/2.0f), sd_i=sqrtf(dm)*sinf(da/2.0f);
    float l1_r=(tr_r+sd_r)/2.0f, l1_i=(tr_i+sd_i)/2.0f;
    float l2_r=(tr_r-sd_r)/2.0f, l2_i=(tr_i-sd_i)/2.0f;
    float l1=sqrtf(l1_r*l1_r+l1_i*l1_i), l2=sqrtf(l2_r*l2_r+l2_i*l2_i);
    float tol=1e-12f;
    float inv_l1=(l1>tol)?1.0f/l1:0.0f, inv_l2=(l2>tol)?1.0f/l2:0.0f;

    float V[2][2][2]={{{1,0},{0,0}},{{0,0},{1,0}}};
    float h01_r=H[0][1][0], h01_i=H[0][1][1];
    float h11_l1_r=H[0][0][0]-l1_r, h11_l1_i=H[0][0][1]-l1_i;
    if (h01_r*h01_r+h01_i*h01_i > 1e-20f) {
        float m1=sqrtf(h01_r*h01_r+h01_i*h01_i+h11_l1_r*h11_l1_r+h11_l1_i*h11_l1_i);
        V[0][0][0]=h01_r/m1; V[0][0][1]=h01_i/m1;
        V[1][0][0]=-h11_l1_r/m1; V[1][0][1]=-h11_l1_i/m1;
    }
    float h11_l2_r=H[0][0][0]-l2_r, h11_l2_i=H[0][0][1]-l2_i;
    if (h01_r*h01_r+h01_i*h01_i > 1e-20f) {
        float m2=sqrtf(h01_r*h01_r+h01_i*h01_i+h11_l2_r*h11_l2_r+h11_l2_i*h11_l2_i);
        V[0][1][0]=h01_r/m2; V[0][1][1]=h01_i/m2;
        V[1][1][0]=-h11_l2_r/m2; V[1][1][1]=-h11_l2_i/m2;
    }
    float Hi[2][2][2];
    for (int i=0;i<2;i++) for (int j=0;j<2;j++) {
        float sr=0,si=0;
        for (int k=0;k<2;k++) {
            float ilk=(k==0)?inv_l1:inv_l2;
            float vr=V[i][k][0]*ilk, vi=V[i][k][1]*ilk;
            float wr=V[j][k][0], wi=-V[j][k][1];
            sr+=vr*wr-vi*wi; si+=vr*wi+vi*wr;
        }
        Hi[i][j][0]=sr; Hi[i][j][1]=si;
    }
    for (int i=0;i<2;i++) for (int j=0;j<m;j++) {
        float sr=0,si=0;
        for (int k=0;k<2;k++) {
            float ar=A[j*2+k].real, ai=-A[j*2+k].imag;
            sr+=Hi[i][k][0]*ar-Hi[i][k][1]*ai;
            si+=Hi[i][k][0]*ai+Hi[i][k][1]*ar;
        }
        pinvA[i*m+j].real=sr; pinvA[i*m+j].imag=si;
    }
}

static void ps_eign_cal(Complex *phi, Complex *ev) {
    float r00=phi[0].real,i00=phi[0].imag;
    float r01=phi[1].real,i01=phi[1].imag;
    float r10=phi[2].real,i10=phi[2].imag;
    float r11=phi[3].real,i11=phi[3].imag;
    float t_r=r00+r11, t_i=i00+i11;
    float det_r=r00*r11-i00*i11-(r01*r10-i01*i10);
    float det_i=r00*i11+i00*r11-(r01*i10+i01*r10);
    float t2_r=t_r*t_r-t_i*t_i, t2_i=2*t_r*t_i;
    float d_r=t2_r-4.0f*det_r, d_i=t2_i-4.0f*det_i;
    float dm=sqrtf(d_r*d_r+d_i*d_i), da=atan2f(d_i,d_r);
    float sd_r=sqrtf(dm)*cosf(da/2.0f), sd_i=sqrtf(dm)*sinf(da/2.0f);
    ev[0].real=(t_r+sd_r)/2.0f; ev[0].imag=(t_i+sd_i)/2.0f;
    ev[1].real=(t_r-sd_r)/2.0f; ev[1].imag=(t_i-sd_i)/2.0f;
}

static void ps_sort_eigenvalues(Complex *eigenvalues, Complex *eigenvectors, int n) {
    for (int i=0;i<n-1;i++) for (int j=0;j<n-i-1;j++) {
        float m1=sqrtf(eigenvalues[j].real*eigenvalues[j].real+eigenvalues[j].imag*eigenvalues[j].imag);
        float m2=sqrtf(eigenvalues[j+1].real*eigenvalues[j+1].real+eigenvalues[j+1].imag*eigenvalues[j+1].imag);
        if (m1<m2) {
            Complex t=eigenvalues[j]; eigenvalues[j]=eigenvalues[j+1]; eigenvalues[j+1]=t;
            for (int r=0;r<n;r++) {
                Complex p=eigenvectors[r*n+j]; eigenvectors[r*n+j]=eigenvectors[r*n+j+1]; eigenvectors[r*n+j+1]=p;
            }
        }
    }
}

static float ps_vel_clip(float v) {
    if (v < -30.0f) return -30.0f;
    if (v >  30.0f) return  30.0f;
    return v;
}

void esprit_ps(const Complex rec_signal[N_SAMPLES], float out[2]) {
    int N=N_SAMPLES, k=2, l=N/2;
    Complex acm[100], eval[10], evec[100];
    ps_spatial_smoothing(rec_signal, N, acm);
    ps_qr_algorithm_eig(acm, l, 10, eval, evec);
    ps_sort_eigenvalues(eval, evec, l);

    int rows_sub=l-1;
    Complex subA[18], subB[18];
    for (int j=0;j<k;j++) for (int i=0;i<rows_sub;i++) {
        subA[i*k+j]=evec[i*l+j];
        subB[i*k+j]=evec[(i+1)*l+j];
    }
    Complex pinvA[18], phi[4];
    ps_svd_pinv_k2(subA, rows_sub, pinvA);

    for (int i=0;i<k;i++) {
        int ir=i*rows_sub;
        for (int j=0;j<k;j++) {
            float sr=0,si=0;
            for (int m=0;m<rows_sub;m++) {
                float pr=pinvA[ir+m].real, pi=pinvA[ir+m].imag;
                float br=subB[m*k+j].real, bi=subB[m*k+j].imag;
                sr+=pr*br-pi*bi; si+=pr*bi+pi*br;
            }
            phi[i*k+j].real=sr; phi[i*k+j].imag=si;
        }
    }
    Complex phi_eigs[2];
    ps_eign_cal(phi, phi_eigs);
    float est[2];
    for (int i=0;i<k;i++) est[i]=-atan2f(phi_eigs[i].imag,phi_eigs[i].real)/0.005f;
    if (est[0]>est[1]) { float t=est[0]; est[0]=est[1]; est[1]=t; }
    out[0]=ps_vel_clip(est[0]);
    out[1]=ps_vel_clip(est[1]);
}

/* ========================================================== */
/*                          Main                              */
/* ========================================================== */
int main(void)
{
    init_platform();
    printf("\n--- ESPRIT PS vs PL Comparison ---\n");

    /* ---- HLS IP init ---- */
    XEsprit_hls        AxiMM;
    XEsprit_hls_Config *cfg = XEsprit_hls_LookupConfig(XPAR_ESPRIT_HLS_0_DEVICE_ID);
    if (!cfg) { xil_printf("Config not found\r\n"); return -1; }
    if (XEsprit_hls_CfgInitialize(&AxiMM, cfg) != XST_SUCCESS) {
        xil_printf("Init failed\r\n"); return -1;
    }

    /* ---- Timing ---- */
    float total_ps_time = 0.0f, total_pl_time = 0.0f;

    /* ---- RMSE accumulators ---- */
    int   count_snr[3]        = {0};
    float sum_sq_err_ps_v1[3] = {0.0f};
    float sum_sq_err_ps_v2[3] = {0.0f};
    float sum_sq_err_pl_v1[3] = {0.0f};
    float sum_sq_err_pl_v2[3] = {0.0f};

    /* ---- Header ---- */
    printf("\n%-5s | %-5s | %-26s | %-26s\n",
           "ID", "SNR", "PS Result (v1, v2)", "PL Result (v1, v2)");
    printf("--------------------------------------------------------------------------\n");

    for (int s = 0; s < NUM_SAMPLES; s++) {

        ESPRIT_Sample *sample = &dataset[s];
        float ps_res[2];
        float pl_input[2 * N_SAMPLES];
        float pl_output[2];
        XTime t0, t1;

        /* ---- PS ---- */
        XTime_SetTime(0); XTime_GetTime(&t0);
        esprit_ps(sample->data, ps_res);
        XTime_GetTime(&t1);
        float ps_time = (float)(t1 - t0) / (COUNTS_PER_SECOND / 1000000);
        total_ps_time += ps_time;

        /* ---- PL ---- */
        for (int i = 0; i < N_SAMPLES; i++) {
            pl_input[2*i]   = sample->data[i].real;
            pl_input[2*i+1] = sample->data[i].imag;
        }
        XTime_SetTime(0); XTime_GetTime(&t0);
        XEsprit_hls_Set_in_mem(&AxiMM,  (u64)pl_input);
        XEsprit_hls_Set_out_mem(&AxiMM, (u64)pl_output);
        XEsprit_hls_Start(&AxiMM);
        while (!XEsprit_hls_IsDone(&AxiMM));
        XTime_GetTime(&t1);
        float pl_time = (float)(t1 - t0) / (COUNTS_PER_SECOND / 1000000);
        total_pl_time += pl_time;

        printf("%-5d | %-5d | (%9.6f, %9.6f)     | (%9.6f, %9.6f)\n",
               sample->sample_id, sample->snr,
               ps_res[0], ps_res[1], pl_output[0], pl_output[1]);

        int snr_idx;
        switch (sample->snr) {
            case -20: snr_idx = 0; break;
            case   0: snr_idx = 1; break;
            case  20: snr_idx = 2; break;
            default:  snr_idx = -1; break;
        }
        if (snr_idx < 0) continue;


        float e_ps1 = sample->gt_vel1 - ps_res[0];
        float e_ps2 = sample->gt_vel2 - ps_res[1];
        float e_pl1 = sample->gt_vel1 - pl_output[0];
        float e_pl2 = sample->gt_vel2 - pl_output[1];

        sum_sq_err_ps_v1[snr_idx] += e_ps1 * e_ps1;
        sum_sq_err_ps_v2[snr_idx] += e_ps2 * e_ps2;
        sum_sq_err_pl_v1[snr_idx] += e_pl1 * e_pl1;
        sum_sq_err_pl_v2[snr_idx] += e_pl2 * e_pl2;
        count_snr[snr_idx]++;
    }

    /* ------------------------------------------------------------------ */
    /* RMSE summary — combined: sqrt((sum_v1 + sum_v2) / (2 * N))         */
    /* ------------------------------------------------------------------ */
    printf("\n--- RMSE per SNR (combined V1+V2, N=%d samples each) ---\n",
           NUM_SAMPLES / 3);
    printf("%-8s | %-14s | %-14s\n", "SNR", "PS RMSE (m/s)", "PL RMSE (m/s)");
    printf("----------------------------------------------\n");

    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            float rmse_ps = sqrtf((sum_sq_err_ps_v1[i] + sum_sq_err_ps_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            float rmse_pl = sqrtf((sum_sq_err_pl_v1[i] + sum_sq_err_pl_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            printf("%-8s | %-14.6f | %-14.6f\n",
                   snr_label_str[i], rmse_ps, rmse_pl);
        }
    }
    printf("----------------------------------------------\n");

    /* ------------------------------------------------------------------ */
    /* Timing summary                                                      */
    /* ------------------------------------------------------------------ */
    printf("\nAverage PS Execution Time : %.4f us\n", total_ps_time / NUM_SAMPLES);
    printf("Average PL Execution Time : %.4f us\n", total_pl_time / NUM_SAMPLES);
    printf("Speedup                   : %.2fx\n",   total_ps_time  / total_pl_time);

    cleanup_platform();
    return 0;
}
