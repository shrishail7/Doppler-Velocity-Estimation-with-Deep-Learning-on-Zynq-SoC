#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <xtime_l.h>
/*
 * ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
 * 
 * This C implementation estimates the Direction of Arrival (DoA) or Doppler velocities
 * for multiple signal sources. This code is developed in reference to the 
 * Python implementation found in 'main.ipynb'.
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
// Helper Functions for Complex Arithmetic (Inlined for Performance)
// ==========================================

// Utility to print complex numbers in 'a + bi' format
static inline void print_complex(Complex c) {
    if (c.imag >= 0)
        printf("%.10f + %.10fi", c.real, c.imag);
    else
        printf("%.10f - %.10fi", c.real, -c.imag);
}

// Basic complex arithmetic functions. These are used throughout the algorithm
// as C does not have native support for complex operations on all platforms.

static inline Complex add_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

static inline Complex sub_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

static inline Complex mul_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

static inline Complex mul_complex_scalar(Complex a, float s) {
    Complex result;
    result.real = a.real * s;
    result.imag = a.imag * s;
    return result;
}

static inline Complex div_complex_scalar(Complex a, float s) {
    Complex result;
    result.real = a.real / s;
    result.imag = a.imag / s;
    return result;
}

static inline Complex conj_complex(Complex a) {
    Complex result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

// -------------------------------------------------------------
// Matrix Operations Helpers
// -------------------------------------------------------------

// Performs matrix multiplication: C = A * B
// Equivalent to Python's '@' operator (e.g., A @ B)
// Assumes all matrices are square of size n x n
void matmul_complex(const Complex *A, const Complex *B, Complex *C, int n) {
    for (int i = 0; i < n; i++) {
        int i_n = i * n;
        for (int j = 0; j < n; j++) {
            double sum_r = 0.0, sum_i = 0.0;
            for (int k = 0; k < n; k++) {
                int k_n = k * n;
                double ar = A[i_n + k].real, ai = A[i_n + k].imag;
                double br = B[k_n + j].real, bi = B[k_n + j].imag;
                sum_r += ar * br - ai * bi;
                sum_i += ar * bi + ai * br;
            }
            C[i_n + j].real = (float)sum_r;
            C[i_n + j].imag = (float)sum_i;
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
            double r_i_r = rec_sig[i].real, r_i_i = rec_sig[i].imag;
            double r_j_r = rec_sig[j].real, r_j_i = rec_sig[j].imag;
            double res_r = r_i_r * r_j_r + r_i_i * r_j_i;
            double res_i = r_i_r * r_j_i - r_i_i * r_j_r;
            autocorrelation_matrix[i * len + j].real = (float)res_r;
            autocorrelation_matrix[i * len + j].imag = (float)res_i;
        }
    }
}

/*
 * Spatial Smoothing
 * Python Reference: Cell 6, internal loop in 'ESPRIT_PS_S_PINV'
 * Python Logic:
 * for i in range(subarr):
 *     sub1 = rec_signal[i:(l + i)]
 *     sub2 = autocorrelation_func_ES(np.array(sub1))
 *     autocorrelation_matrix += sub2
 * autocorrelation_matrix /= subarr
 * 
 * Logic: Subdivides the receiver array into overlapping subarrays (L) to decorrelate signals.
 * This increases the rank of the covariance matrix, especially useful in coherent environments.
 */
void spatial_smoothing(const Complex *rec_signal, int N, Complex *output_matrix) {
    int l = N / 2; // length of subarray
    int subarr = N + 1 - l; // number of subarrays

    double accum_r[100] = {0}; // Max size L=10 -> 10*10=100
    double accum_i[100] = {0};

    for (int i = 0; i < subarr; i++) {
        const Complex *sig_ptr = &rec_signal[i];
        for(int j=0; j<l; j++) {
            int j_l = j * l;
            double r_j_r = sig_ptr[j].real, r_j_i = sig_ptr[j].imag;
            for(int k=0; k<l; k++) {
                double r_k_r = sig_ptr[k].real, r_k_i = sig_ptr[k].imag;
                accum_r[j_l + k] += r_j_r * r_k_r + r_j_i * r_k_i;
                accum_i[j_l + k] += r_j_r * r_k_i - r_j_i * r_k_r;
            }
        }
    }

    float inv_subarr = 1.0f / (float)subarr;
    for(int k=0; k<l*l; k++) {
        output_matrix[k].real = (float)(accum_r[k] * inv_subarr);
        output_matrix[k].imag = (float)(accum_i[k] * inv_subarr);
    }
}

/*
 * Autocovariance Matrix Calculation
 * Python Reference: Cell 5, function 'autocorrelation_func_ES(rec_sig)'
 * Python Logic:
 * def autocorrelation_func_ES(rec_sig):
 *     conj_sig = np.conj(rec_sig)
 *     autocorrelation_matrix = []
 *     for i in range(len(conj_sig)):
 *         dat1 = conj_sig[i] * rec_sig
 *         autocorrelation_matrix.append(dat1.astype(np.complex64))
 *     return np.array(autocorrelation_matrix,dtype = np.complex64)
 * 
 * Logic: Computes the outer product of the signal vector with its conjugate
 * to form the Rxx matrix. Rxx = E[x * x^H]
 */

/*
 * Givens Rotation Calculation
 * Python Reference: Cell 5, function 'givensrotation_new(a, b)'
 * Python Logic:
 * def givensrotation_new(a, b):
 *     hypo=np.sqrt(a.real**2 + a.imag**2 + b.real**2 + b.imag**2) 
 *     if hypo == 0:
 *         return 1, 0
 *     else:
 *         cos1 = a/ hypo
 *         sin1 = b / hypo 
 *         return cos1, sin1
 * 
 * Logic: Calculates the sine and cosine values to zero out elements below the diagonal during QR.
 */
void givensrotation_new(Complex a, Complex b, Complex *cos_out, Complex *sin_out) {
    double a_r = a.real, a_i = a.imag, b_r = b.real, b_i = b.imag;
    double hypo = sqrt(a_r*a_r + a_i*a_i + b_r*b_r + b_i*b_i);
    if (hypo == 0.0) {
        cos_out->real = 1.0f; cos_out->imag = 0.0f;
        sin_out->real = 0.0f; sin_out->imag = 0.0f;
    } else {
        cos_out->real = (float)(a_r / hypo);
        cos_out->imag = (float)(a_i / hypo);
        sin_out->real = (float)(b_r / hypo);
        sin_out->imag = (float)(b_i / hypo);
    }
}

/*
 * QR Decomposition using Givens Rotations
 * Python Reference: Cell 5, function 'qr_givens_S(A)'
 * Python Logic:
 * def qr_givens_S(A):
 *     m, n = A.shape
 *     R = A.copy()
 *     Q = np.identity(m, dtype=np.complex64)
 *     for i in range(0, n - 1):
 *         for j in range(i + 1, m):
 *             cos, sin = givensrotation_new(R[i, i], R[j, i])
 *             R[i], R[j] = (R[i]*np.conj(cos)) + (R[j]*np.conj(sin)), (R[i]*(-sin)) + (R[j] * cos)
 *             Q[i], Q[j] = (Q[i]*np.conj(cos)) + (Q[j]*np.conj(sin)), (Q[i]*(-sin)) + (Q[j] * cos)
 *     return np.transpose(np.conj(Q)), R
 * 
 * Logic: Orthogonalizes the matrix into Q (orthogonal) and R (upper triangular).
 * A = Q * R
 */
void qr_givens_S(const Complex *A, int n, Complex *Q_out, Complex *R_out) {
    // 1. Initialize R as a copy of A, and Q_temp as an Identity matrix
    copy_matrix((Complex *)A, R_out, n);
    Complex Q_temp[100]; // Max size n=10 -> 100
    identity_matrix(Q_temp, n);

    // 2. Iterate through columns (i) and rows below diagonal (j)
    for (int i = 0; i < n - 1; i++) {
        int i_n = i * n;
        for (int j = i + 1; j < n; j++) {
            int j_n = j * n;
            
            // 3. Compute Givens rotation to zero out R[j, i] using R[i, i]
            Complex cos, sin;
            givensrotation_new(R_out[i_n + i], R_out[j_n + i], &cos, &sin);

            // Pre-calculate real and imaginary parts for rotation arithmetic
            double c_r = cos.real, c_i = cos.imag;
            double s_r = sin.real, s_i = sin.imag;
            double conj_c_r = c_r, conj_c_i = -c_i;
            double conj_s_r = s_r, conj_s_i = -s_i;
            double neg_s_r = -s_r, neg_s_i = -s_i;

            // 4. Apply rotations to rows i and j of R and Q
            // This is equivalent to multiplying by the Givens matrix G from the left
            for (int k = 0; k < n; k++) {
                // Update R matrix: R = G * R
                double Ri_r = R_out[i_n + k].real, Ri_i = R_out[i_n + k].imag;
                double Rj_r = R_out[j_n + k].real, Rj_i = R_out[j_n + k].imag;
                
                // Row i update: Ri = conj(c)*Ri + conj(s)*Rj
                double t1_r = Ri_r*conj_c_r - Ri_i*conj_c_i + Rj_r*conj_s_r - Rj_i*conj_s_i;
                double t1_i = Ri_r*conj_c_i + Ri_i*conj_c_r + Rj_r*conj_s_i + Rj_i*conj_s_r;
                
                // Row j update: Rj = -s*Ri + c*Rj (zeroes out R[j, i] on the first k=i iteration)
                double t2_r = Ri_r*neg_s_r - Ri_i*neg_s_i + Rj_r*c_r - Rj_i*c_i;
                double t2_i = Ri_r*neg_s_i + Ri_i*neg_s_r + Rj_r*c_i + Rj_i*c_r;
                
                R_out[i_n + k].real = (float)t1_r; R_out[i_n + k].imag = (float)t1_i;
                R_out[j_n + k].real = (float)t2_r; R_out[j_n + k].imag = (float)t2_i;

                // Update Q matrix (stored as Q^H): Q^H = G * Q^H
                double Qi_r = Q_temp[i_n + k].real, Qi_i = Q_temp[i_n + k].imag;
                double Qj_r = Q_temp[j_n + k].real, Qj_i = Q_temp[j_n + k].imag;
                
                double qt1_r = Qi_r*conj_c_r - Qi_i*conj_c_i + Qj_r*conj_s_r - Qj_i*conj_s_i;
                double qt1_i = Qi_r*conj_c_i + Qi_i*conj_c_r + Qj_r*conj_s_i + Qj_i*conj_s_r;
                
                double qt2_r = Qi_r*neg_s_r - Qi_i*neg_s_i + Qj_r*c_r - Qj_i*c_i;
                double qt2_i = Qi_r*neg_s_i + Qi_i*neg_s_r + Qj_r*c_i + Qj_i*c_r;
                
                Q_temp[i_n + k].real = (float)qt1_r; Q_temp[i_n + k].imag = (float)qt1_i;
                Q_temp[j_n + k].real = (float)qt2_r; Q_temp[j_n + k].imag = (float)qt2_i;
            }
        }
    }
    // 5. Final Q = (Q^H)^H. Inverts the accumulation to get the orthogonal matrix Q
    transpose_conj_matrix(Q_temp, Q_out, n);
}

/*
 * Eigenvalue Decomposition (Iterative QR Algorithm)
 * Python Reference: Cell 5, function 'qr_algorithm_eig_single(A_input, max_iterations=20)'
 * Python Logic:
 * def qr_algorithm_eig_single(A_input, max_iterations=20):  
 *     A = np.array(A_input, dtype=np.complex64, copy=True)
 *     n = A.shape[0]    
 *     Q_total = np.eye(n, dtype=np.complex64)
 *     for iteration in range(max_iterations):
 *         Q, R = qr_givens_S(A)
 *         A = (R @ Q).astype(np.complex64)        
 *         Q_total = (Q_total @ Q).astype(np.complex64)    
 *     eigenvalues = np.diag(A).astype(np.complex64)
 *     eigenvectors = Q_total.astype(np.complex64)    
 *     return eigenvalues, eigenvectors
 * 
 * Logic: Uses the QR algorithm shift to converge A towards a diagonal matrix of eigenvalues.
 * Simultaneously updates the 'eigenvectors' matrix to obtain the signal/noise subspace.
 */
void qr_algorithm_eig_single(const Complex *A_input, int n, int max_iterations, Complex *eigenvalues, Complex *eigenvectors) {
    Complex A[100], Q[100], R[100], TempA[100], TempQTotal[100];
    copy_matrix((Complex *)A_input, A, n);
    identity_matrix(eigenvectors, n);

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
}

// SVD-based Pseudo-Inverse (Optimized for 2x2 Phi Matrix)
// Python Reference: Cell 6, internal call to np.linalg.pinv(subA1)
// Logic: Specifically calculates the Moore-Penrose inverse of the first signal subspace partition.
// This is critical for Total Least Squares (TLS) ESPRIT.
void svd_pinv_complex_k2(Complex *A, int m, Complex *pinvA) {
    // 1. Compute the Gram matrix H = A^H * A (size 2x2)
    // This reduces the problem to an eigenvalue problem for the Hermitian matrix H.
    double H[2][2][2] = {{{0}}}; // H[row][col][0=real, 1=imag]
    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            double sum_r = 0, sum_i = 0;
            for(int k=0; k<m; ++k) {
                // Conjugate transpose lookup for A^H
                double a_ik_r = A[k*2+i].real, a_ik_i = -A[k*2+i].imag; 
                double a_kj_r = A[k*2+j].real, a_kj_i = A[k*2+j].imag;
                sum_r += a_ik_r * a_kj_r - a_ik_i * a_kj_i;
                sum_i += a_ik_r * a_kj_i + a_ik_i * a_kj_r;
            }
            H[i][j][0] = sum_r;
            H[i][j][1] = sum_i;
        }
    }

    // 2. Calculate the Trace and Determinant of H for eigenvalue solving
    double trace_r = H[0][0][0] + H[1][1][0];
    double trace_i = H[0][0][1] + H[1][1][1]; 
    
    double ad_r = H[0][0][0] * H[1][1][0] - H[0][0][1] * H[1][1][1];
    double ad_i = H[0][0][0] * H[1][1][1] + H[0][0][1] * H[1][1][0];
    double bc_r = H[0][1][0] * H[1][0][0] - H[0][1][1] * H[1][0][1];
    double bc_i = H[0][1][0] * H[1][0][1] + H[0][1][1] * H[1][0][0];
    
    double det_r = ad_r - bc_r;
    double det_i = ad_i - bc_i;

    // 3. Solve the characteristic equation: lambda^2 - trace*lambda + det = 0
    // Using the quadratic formula with complex support
    double tr_sq_r = trace_r*trace_r - trace_i*trace_i;
    double tr_sq_i = 2*trace_r*trace_i;
    double disc_r = tr_sq_r - 4*det_r;
    double disc_i = tr_sq_i - 4*det_i;
    
    double disc_mag = sqrt(disc_r*disc_r + disc_i*disc_i);
    double disc_ang = atan2(disc_i, disc_r);
    
    double sqrt_disc_r = sqrt(disc_mag) * cos(disc_ang / 2.0);
    double sqrt_disc_i = sqrt(disc_mag) * sin(disc_ang / 2.0);
    
    double lambda1_r = (trace_r + sqrt_disc_r) / 2.0;
    double lambda1_i = (trace_i + sqrt_disc_i) / 2.0;
    double lambda2_r = (trace_r - sqrt_disc_r) / 2.0;
    double lambda2_i = (trace_i - sqrt_disc_i) / 2.0;
    
    // Singular values are the square roots of the eigenvalues of H
    double l1 = sqrt(lambda1_r*lambda1_r + lambda1_i*lambda1_i);
    double l2 = sqrt(lambda2_r*lambda2_r + lambda2_i*lambda2_i);
    
    double tol = 1e-12;
    double inv_l1 = (l1 > tol) ? 1.0 / l1 : 0.0;
    double inv_l2 = (l2 > tol) ? 1.0 / l2 : 0.0;
    
    // 4. Find the Eigenvectors V of H
    double V[2][2][2] = {{{1, 0}, {0, 0}}, {{0, 0}, {1, 0}}};
    double h11_l1_r = H[0][0][0] - lambda1_r;
    double h11_l1_i = H[0][0][1] - lambda1_i;
    
    if (H[0][1][0]*H[0][1][0] + H[0][1][1]*H[0][1][1] > 1e-20) {
        double mag1 = sqrt(H[0][1][0]*H[0][1][0] + H[0][1][1]*H[0][1][1] + h11_l1_r*h11_l1_r + h11_l1_i*h11_l1_i);
        V[0][0][0] = H[0][1][0]/mag1; V[0][0][1] = H[0][1][1]/mag1;
        V[1][0][0] = -h11_l1_r/mag1; V[1][0][1] = -h11_l1_i/mag1;
    }
    
    double h11_l2_r = H[0][0][0] - lambda2_r;
    double h11_l2_i = H[0][0][1] - lambda2_i;
    if (H[0][1][0]*H[0][1][0] + H[0][1][1]*H[0][1][1] > 1e-20) {
        double mag2 = sqrt(H[0][1][0]*H[0][1][0] + H[0][1][1]*H[0][1][1] + h11_l2_r*h11_l2_r + h11_l2_i*h11_l2_i);
        V[0][1][0] = H[0][1][0]/mag2; V[0][1][1] = H[0][1][1]/mag2;
        V[1][1][0] = -h11_l2_r/mag2; V[1][1][1] = -h11_l2_i/mag2;
    }
    
    // 5. Compute the Inverse of H: H_inv = V * inv(Sigma) * V^H
    double H_inv[2][2][2] = {{{0}}};
    for(int i=0; i<2; ++i) {
        for(int j=0; j<2; ++j) {
            double sum_r = 0, sum_i = 0;
            for(int k=0; k<2; ++k) {
                double inv_lk = (k == 0) ? inv_l1 : inv_l2;
                double vik_r = V[i][k][0] * inv_lk, vik_i = V[i][k][1] * inv_lk;
                double vhkj_r = V[j][k][0], vhkj_i = -V[j][k][1];
                sum_r += vik_r * vhkj_r - vik_i * vhkj_i;
                sum_i += vik_r * vhkj_i + vik_i * vhkj_r;
            }
            H_inv[i][j][0] = sum_r; H_inv[i][j][1] = sum_i;
        }
    }
    
    // 6. Final Pseudo-Inverse: A_pinv = H_inv * A^H
    for(int i=0; i<2; ++i) {
        for(int j=0; j<m; ++j) {
            double sum_r = 0, sum_i = 0;
            for(int k=0; k<2; ++k) {
                double ah_kj_r = A[j*2+k].real, ah_kj_i = -A[j*2+k].imag;
                sum_r += H_inv[i][k][0] * ah_kj_r - H_inv[i][k][1] * ah_kj_i;
                sum_i += H_inv[i][k][0] * ah_kj_i + H_inv[i][k][1] * ah_kj_r;
            }
            pinvA[i*m+j].real = (float)sum_r;
            pinvA[i*m+j].imag = (float)sum_i;
        }
    }
}

/*
 * Eigenvalue Calculation for the Phi Matrix
 * Python Reference: Cell 5, function 'eign_cal_ES(phi_mat)'
 * Python Logic:
 * def eign_cal_ES(phi_mat):    
 *     trace = phi_mat[0,0] + phi_mat[1,1]
 *     determinant = phi_mat[0, 0] * phi_mat[1, 1] - phi_mat[0, 1] * phi_mat[1, 0]
 *     discriminant = trace**2 - np.complex64(4.0) * determinant
 *     lambda1 = (trace + np.sqrt(discriminant, dtype=np.complex64)) / np.complex64(2.0)
 *     lambda2 = (trace - np.sqrt(discriminant, dtype=np.complex64)) / np.complex64(2.0)
 *     return np.array([lambda1, lambda2], dtype=np.complex64)
 * 
 * Logic: Solves the characteristic equation of the 2x2 Phi matrix: det(Phi - lambda * I) = 0.
 */
void eign_cal_ES(Complex *phi_mat, Complex *eigenvalues) {
    // Extracts real and imaginary parts of the 2x2 Phi matrix
    double r00 = phi_mat[0].real, i00 = phi_mat[0].imag;
    double r01 = phi_mat[1].real, i01 = phi_mat[1].imag;
    double r10 = phi_mat[2].real, i10 = phi_mat[2].imag;
    double r11 = phi_mat[3].real, i11 = phi_mat[3].imag;
    
    // 1. Compute Trace (sum of diagonal elements)
    double t_r = r00 + r11;
    double t_i = i00 + i11;
    
    // 2. Compute Determinant (ad - bc)
    double det_r = r00 * r11 - i00 * i11 - (r01 * r10 - i01 * i10);
    double det_i = r00 * i11 + i00 * r11 - (r01 * i10 + i01 * r10);
    
    // 3. Solve characteristic equation: lambda^2 - trace*lambda + determinant = 0
    double t2_r = t_r * t_r - t_i * t_i;
    double t2_i = 2 * t_r * t_i;
    
    double disc_r = t2_r - 4.0 * det_r;
    double disc_i = t2_i - 4.0 * det_i;
    
    // 4. Calculate discriminant magnitude and angle for complex square root
    double disc_mag = sqrt(disc_r * disc_r + disc_i * disc_i);
    double disc_ang = atan2(disc_i, disc_r);
    
    double sqrt_disc_r = sqrt(disc_mag) * cos(disc_ang / 2.0);
    double sqrt_disc_i = sqrt(disc_mag) * sin(disc_ang / 2.0);
    
    // 5. Final Eigenvalues: lambda = (trace +/- sqrt(discriminant)) / 2
    eigenvalues[0].real = (float)((t_r + sqrt_disc_r) / 2.0);
    eigenvalues[0].imag = (float)((t_i + sqrt_disc_i) / 2.0);
    
    eigenvalues[1].real = (float)((t_r - sqrt_disc_r) / 2.0);
    eigenvalues[1].imag = (float)((t_i - sqrt_disc_i) / 2.0);
}

// Ensures signal subspace corresponds to most dominant components
// Logic: Sorts eigenvalues by magnitude and swaps corresponding columns in the eigenvector matrix.
// This is required to isolate the signal subspace (the 'k' largest eigenvalues).
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
      * Batch Processing Loop
      * Python Reference: Cell 9
      * Logic: Processes each sample in the provided dataset, calculates velocities,
      * and compares them against ground truth (GT) to compute RMSE.
      */
     // Variables for RMSE calculation
     float snr_sq_err[3] = {0.0f, 0.0f, 0.0f};
     int snr_processed_count[3] = {0, 0, 0};
     int snr_list[3] = {-20, 0, 20};

     for (int s = 0; s < NUM_SAMPLES; s++) {
         const ESPRIT_Sample *current = &dataset[s];
         int N = N_SAMPLES;
         int k = 2; // Signal subspace dimension (M in Python)
         int l = N / 2; // Subarray length (L in Python)

         printf("\n>>> PROCESSING: SNR %d, Sample ID %d <<<\n", current->snr, current->sample_id);
         printf("Ground Truth: %.10f, %.10f\n", current->gt_vel1, current->gt_vel2);

         // 1. Spatial Smoothing (Cell 6 logic: ESPRIT_PS_S_PINV)
         // Logic: Computes the averaged autocovariance matrix across subarrays.
         Complex autocorrelation_matrix[100]; // L=10 -> 100
         spatial_smoothing(current->data, N, autocorrelation_matrix);

         // 2. Eigenvalue Decomposition (Cell 6: qr_algorithm_eig_single)
         // Logic: Finds the principal components of the covariance matrix.
         Complex eigenvalues_all[10]; // L=10
         Complex eigenvectors_all[100]; // L*L=100
         qr_algorithm_eig_single(autocorrelation_matrix, l, 20, eigenvalues_all, eigenvectors_all);

         // 3. Sorting (Implicit in Notebook, Explicit here for stability)
         sort_eigenvalues(eigenvalues_all, eigenvectors_all, l);

         // 4. Sub-matrices (Cell 6: S extract and split into S1 and S2)
         // Python Logic:
         // S = eign_vec[:, :k]
         // subA1 = S[0 : autocorrelation_matrix.shape[0] - 1, :]
         // subB1 = S[1 : autocorrelation_matrix.shape[0], :]
         // Logic: S is the signal subspace. S1 (subA1) is S without the last row, 
         // and S2 (subB1) is S without the first row. The shift invariance property
         // implies S2 = S1 * Phi.
         int rows_sub = l - 1;
         Complex subA1[18]; // (L-1)*k = 9*2 = 18
         Complex subB1[18];
         for(int j=0; j<k; j++) {
             for(int i=0; i<rows_sub; i++) {
                 subA1[i*k+j] = eigenvectors_all[i*l+j];
                 subB1[i*k+j] = eigenvectors_all[(i+1)*l+j];
             }
         }

         // 5. Phi Matrix (Cell 6: phi_mat = pinv(subA1) @ subB1)
         // Logic: Solves the Total Least Squares problem to find the rotational 
         // operator 'Phi' that maps S1 to S2.
         Complex pinvA1[18]; // k*(L-1) = 2*9 = 18
         svd_pinv_complex_k2(subA1, rows_sub, pinvA1);
         Complex phi_mat[4];
         
         // Perform Complex Matrix Multiplication: phi_mat = pinvA1 * subB1
         // Dimensions: pinvA1 [k x rows_sub]  @  subB1 [rows_sub x k]  =>  phi_mat [k x k]
         for(int i=0; i<k; i++) {
             int i_rows = i * rows_sub; // Row offset for pinvA1
             for(int j=0; j<k; j++) {
                 double sum_r = 0.0, sum_i = 0.0;
                 // Inner dot product of pinvA1's row and subB1's column
                 for(int m=0; m<rows_sub; m++) {
                     // pinvA1[i, m] * subB1[m, j]
                     double pr = pinvA1[i_rows + m].real, pi = pinvA1[i_rows + m].imag;
                     double br = subB1[m*k+j].real, bi = subB1[m*k+j].imag;
                     
                     // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                     sum_r += pr * br - pi * bi;
                     sum_i += pr * bi + pi * br;
                 }
                 phi_mat[i*k+j].real = (float)sum_r;
                 phi_mat[i*k+j].imag = (float)sum_i;
             }
         }

         // 6. Eigenvalues of Phi & Doppler estimation (Cell 6: phi = eign_cal_ES(phi_mat))
         // Logic: The angles of the eigenvalues of Phi correspond to the 
         // normalized Doppler frequencies.
         Complex phi_eigs[2];
         eign_cal_ES(phi_mat, phi_eigs);
         float estimates[2];
         for(int i=0; i<k; i++) {
             float angle = atan2f(phi_eigs[i].imag, phi_eigs[i].real);
             estimates[i] = -angle / 0.005f; // Scaling to velocity (m/s)
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
         float gt[2] = {(float)current->gt_vel1, (float)current->gt_vel2};
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
             snr_sq_err[snr_idx] += (clipped0 - gt[0]) * (clipped0 - gt[0]);
             snr_sq_err[snr_idx] += (clipped1 - gt[1]) * (clipped1 - gt[1]);
             snr_processed_count[snr_idx]++;
         }
     }

 	// XTime_GetTime(&esprit_end);



 	// total_time = (float)1.0 * (esprit_end -esprit_start) / (COUNTS_PER_SECOND / 1000000);
 	// printf("\n######## Execution Time ####\n");
 	// printf("PS Execution Time for ESPRIT (micro-sec) %f \n", total_time/15);


     /*
      * RMSE Results Calculation and Display
      * Formula: RMSE = sqrt( sum( (estimate[i] - ground_truth[i])^2 ) / (N_samples * M) )
      * where M is the number of signal sources (k=2).
      */
     printf("\n========================================\n");
     printf("   RMSE RESULTS FOR EACH SNR\n");
     printf("========================================\n");
     for(int i=0; i<3; i++) {
         if(snr_processed_count[i] > 0) {
             // Total squared error is divided by the total number of estimates 
             // (processed_samples * sources_per_sample) before taking the square root.
             float rmse = sqrtf(snr_sq_err[i] / (snr_processed_count[i] * 2.0f));
             printf("SNR %4d: RMSE = %.10f\n", snr_list[i], rmse);
         }
     }

     printf("\n========================================\n");
     printf("   BATCH PROCESSING COMPLETE            \n");
     printf("========================================\n");

     return 0;
 }