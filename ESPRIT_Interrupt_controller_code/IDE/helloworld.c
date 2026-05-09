#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xaxidma.h"
#include "xparameters.h"
#include "xtime_l.h"
#include <math.h>
// Define Complex structure used in inputs.h and PS logic
typedef struct {
    float real;
    float imag;
} Complex;

#include "inputs.h"

#include "xscugic.h" // 1
#define RESET_TIMEOUT_COUNTER	10000

XScuGic INTCInst; //2
/*
 * Flags interrupt handlers use to notify the application context the events.
 */
volatile int MM2SDone;
volatile int S2MMDone;
volatile int Error;

// ISR 3
static void MM2SIntrHandler(void *Callback)
{

	u32 IrqStatus;
	int TimeOut;
	XAxiDma *AxiDmaInst = (XAxiDma *)Callback;
	/* Read pending interrupts */
	IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DMA_TO_DEVICE);

	/* Acknowledge pending interrupts */
	XAxiDma_IntrAckIrq(AxiDmaInst, IrqStatus, XAXIDMA_DMA_TO_DEVICE);

	//Check whether correct DMA has raised the interrupt
	if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK))
	{
		return;
	}

	/*
	 * If error interrupt is asserted, raise error flag, reset the
	 * hardware to recover from the error, and return with no further
	 * processing.
	 */
	if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK))
	{
		Error = 1;
		 // Reset should never fail for transmit channel
		XAxiDma_Reset(AxiDmaInst);
		TimeOut = RESET_TIMEOUT_COUNTER;
		while (TimeOut)
		{
			if (XAxiDma_ResetIsDone(AxiDmaInst))
			{
				break;
			}
			TimeOut -= 1;
		}
		return;
	}
    // If Completion interrupt is asserted, then set the MM2SDone flag
	if ((IrqStatus & XAXIDMA_IRQ_IOC_MASK))
	{
		MM2SDone = 1;
	}
}


//ISR 4
static void S2MMIntrHandler(void *Callback)
{
	u32 IrqStatus;
	int TimeOut;
	XAxiDma *AxiDmaInst = (XAxiDma *)Callback;

	/* Read pending interrupts */
	IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DEVICE_TO_DMA);

	/* Acknowledge pending interrupts */
	XAxiDma_IntrAckIrq(AxiDmaInst, IrqStatus, XAXIDMA_DEVICE_TO_DMA);

	//Check whether correct DMA has raised the interrupt
	if (!(IrqStatus & XAXIDMA_IRQ_ALL_MASK))
	{
		return;
	}

	/*
	 * If error interrupt is asserted, raise error flag, reset the
	 * hardware to recover from the error, and return with no further
	 * processing.
	 */
	if ((IrqStatus & XAXIDMA_IRQ_ERROR_MASK))
	{
		Error = 1;
		/* Reset could fail and hang
		 * NEED a way to handle this or do not call it??
		 */
		XAxiDma_Reset(AxiDmaInst);
		TimeOut = RESET_TIMEOUT_COUNTER;
		while (TimeOut)
		{
			if(XAxiDma_ResetIsDone(AxiDmaInst))
			{
				break;
			}
			TimeOut -= 1;
		}
		return;
	}

	 // If completion interrupt is asserted, then set S2MMDone flag
	if ((IrqStatus & XAXIDMA_IRQ_IOC_MASK))
	{
		S2MMDone = 1;
	}
}



// INTR Setup 5
static int SetupIntrSystem(XScuGic * IntcInstancePtr,  XAxiDma * AxiDmaPtr, u16 MM2SIntrId, u16 S2MMIntrId)
{
	int Status;
	XScuGic_Config *IntcConfig;

	IntcConfig = XScuGic_LookupConfig(XPAR_PS7_SCUGIC_0_DEVICE_ID);
	if (NULL == IntcConfig)
	{
		return XST_FAILURE;
	}

	Status = XScuGic_CfgInitialize(IntcInstancePtr, IntcConfig, IntcConfig->CpuBaseAddress);
	if (Status != XST_SUCCESS)
	{
		return XST_FAILURE;
	}

   // Initialize Exception handling on the ARM processor
	Xil_ExceptionInit();

	// Connect the supplied Xilinx general interrupt handler
	// to the interrupt handling logic in the processor.
	// All interrupts go through the interrupt controller, so the
	// ARM processor has to first "ask" the interrupt controller
	// which peripheral generated the interrupt.  The handler that
	// does this is supplied by Xilinx and is called "XScuGic_InterruptHandler"
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,(Xil_ExceptionHandler)XScuGic_InterruptHandler, (void *)IntcInstancePtr);


	/*
	 * Connect the device driver handler that will be called when an
	 * interrupt for the device occurs, the handler defined above performs
	 * the specific interrupt processing for the device.
	 */
	// Assign (connect) our interrupt handler
	Status = XScuGic_Connect(IntcInstancePtr, MM2SIntrId, (Xil_InterruptHandler)MM2SIntrHandler, AxiDmaPtr);
	if (Status != XST_SUCCESS) {
		return Status;
	}

	Status = XScuGic_Connect(IntcInstancePtr, S2MMIntrId, (Xil_InterruptHandler)S2MMIntrHandler, AxiDmaPtr);
	if (Status != XST_SUCCESS) {
		return Status;
	}

	// Enable the interrupt *input* on the GIC for the DMA interrupt
	XScuGic_Enable(IntcInstancePtr, MM2SIntrId);
	XScuGic_Enable(IntcInstancePtr, S2MMIntrId);

	XScuGic_SetPriorityTriggerType(IntcInstancePtr, MM2SIntrId, 0xA0, 0x3);
	XScuGic_SetPriorityTriggerType(IntcInstancePtr, S2MMIntrId, 0xA0, 0x3);

	/* Enable all interrupts */
	XAxiDma_IntrEnable(AxiDmaPtr, XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_IntrEnable(AxiDmaPtr, XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DEVICE_TO_DMA);

	// Enable interrupts in the ARM Processor.
	Xil_ExceptionEnable();

	return XST_SUCCESS;
}


static void DisconnIntrSystem(XScuGic * IntcInstancePtr, u16 MM2SIntrId, u16 S2MMIntrId)
{
	XScuGic_Disconnect(IntcInstancePtr, MM2SIntrId);
	XScuGic_Disconnect(IntcInstancePtr, S2MMIntrId);
}

// Forward declarations to avoid implicit declaration warnings
static inline Complex ps_conj_complex(Complex a);
static void ps_matmul_complex(const Complex *A, const Complex *B, Complex *C, int n);
static void ps_transpose_conj_matrix(Complex *A, Complex *B, int n);
static void ps_copy_matrix(Complex *src, Complex *dst, int n);
static void ps_identity_matrix(Complex *A, int n);
static void ps_spatial_smoothing(const Complex *rec_signal, int N, Complex *output_matrix);
static void ps_givensrotation(Complex a, Complex b, Complex *cos_out, Complex *sin_out);
static void ps_qr_givens(const Complex *A, int n, Complex *Q_out, Complex *R_out);
static void ps_qr_algorithm_eig(const Complex *A_input, int n, int max_iterations, Complex *eigenvalues, Complex *eigenvectors);
static void ps_svd_pinv_k2(Complex *A, int m, Complex *pinvA);
static void ps_eign_cal(Complex *phi_mat, Complex *eigenvalues);
static void ps_sort_eigenvalues(Complex *eigenvalues, Complex *eigenvectors, int n);
static float ps_vel_clip(float v);
void esprit_ps(const Complex rec_signal[N_SAMPLES], float out_estimates[2]);


// --- PS (Software)--- //
static inline Complex ps_conj_complex(Complex a) {
    Complex result;
    result.real = a.real;
    result.imag = -a.imag;
    return result;
}

static void ps_matmul_complex(const Complex *A, const Complex *B, Complex *C, int n) {
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

static void ps_transpose_conj_matrix(Complex *A, Complex *B, int n) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            B[j*n+i] = ps_conj_complex(A[i*n+j]);
        }
    }
}

static void ps_copy_matrix(Complex *src, Complex *dst, int n) {
    for(int i=0; i<n*n; i++) {
        dst[i] = src[i];
    }
}

static void ps_identity_matrix(Complex *A, int n) {
    for(int i=0; i<n*n; i++) {
        A[i].real = 0.0f; A[i].imag = 0.0f;
    }
    for(int i=0; i<n; i++) {
        A[i*n+i].real = 1.0f;
    }
}

static void ps_spatial_smoothing(const Complex *rec_signal, int N, Complex *output_matrix) {
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

static void ps_givensrotation(Complex a, Complex b, Complex *cos_out, Complex *sin_out) {
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

static void ps_qr_givens(const Complex *A, int n, Complex *Q_out, Complex *R_out) {
    ps_copy_matrix((Complex *)A, R_out, n);
    Complex Q_temp[100];
    ps_identity_matrix(Q_temp, n);

    for (int i = 0; i < n - 1; i++) {
        int i_n = i * n;
        for (int j = i + 1; j < n; j++) {
            int j_n = j * n;
            Complex cosf, sinf;
            ps_givensrotation(R_out[i_n + i], R_out[j_n + i], &cosf, &sinf);

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
    ps_transpose_conj_matrix(Q_temp, Q_out, n);
}

static void ps_qr_algorithm_eig(const Complex *A_input, int n, int max_iterations, Complex *eigenvalues, Complex *eigenvectors) {
    Complex A[100], Q[100], R[100], TempA[100], TempQTotal[100];
    ps_copy_matrix((Complex *)A_input, A, n);
    ps_identity_matrix(eigenvectors, n);

    for (int iter = 0; iter < max_iterations; iter++) {
        ps_qr_givens(A, n, Q, R);
        ps_matmul_complex(R, Q, TempA, n);
        ps_copy_matrix(TempA, A, n);
        ps_matmul_complex(eigenvectors, Q, TempQTotal, n);
        ps_copy_matrix(TempQTotal, eigenvectors, n);
    }
    for(int i=0; i<n; i++) {
        eigenvalues[i] = A[i*n+i];
    }
}

static void ps_svd_pinv_k2(Complex *A, int m, Complex *pinvA) {
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

static void ps_eign_cal(Complex *phi_mat, Complex *eigenvalues) {
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

static void ps_sort_eigenvalues(Complex *eigenvalues, Complex *eigenvectors, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            float mag1 = sqrtf(eigenvalues[j].real*eigenvalues[j].real + eigenvalues[j].imag*eigenvalues[j].imag);
            float mag2 = sqrtf(eigenvalues[j+1].real*eigenvalues[j+1].real + eigenvalues[j+1].imag*eigenvalues[j+1].imag);
            if (mag1 < mag2) {
                Complex tV = eigenvalues[j]; eigenvalues[j] = eigenvalues[j+1]; eigenvalues[j+1] = tV;
                for (int r = 0; r < n; r++) {
                    Complex tP = eigenvectors[r*n+j]; eigenvectors[r*n+j] = eigenvectors[r*n+j+1]; eigenvectors[r*n+j+1] = tP;
                }
            }
        }
    }
}

static float ps_vel_clip(float v) {
    if (v < -30.0f) return -30.0f;
    if (v > 30.0f) return 30.0f;
    return v;
}

void esprit_ps(const Complex rec_signal[N_SAMPLES], float out_estimates[2]) {
    int N = N_SAMPLES;
    int k = 2;
    int l = N / 2;

    Complex autocorrelation_matrix[100];
    ps_spatial_smoothing(rec_signal, N, autocorrelation_matrix);

    Complex eigenvalues_all[10];
    Complex eigenvectors_all[100];
    ps_qr_algorithm_eig(autocorrelation_matrix, l, 10, eigenvalues_all, eigenvectors_all);

    ps_sort_eigenvalues(eigenvalues_all, eigenvectors_all, l);

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
    ps_svd_pinv_k2(subA1, rows_sub, pinvA1);
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
    ps_eign_cal(phi_mat, phi_eigs);
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

    out_estimates[0] = ps_vel_clip(estimates[0]);
    out_estimates[1] = ps_vel_clip(estimates[1]);
}



/* SNR label strings matching switch indices 0->-20dB, 1->0dB, 2->20dB */
static const char *snr_label_str[3] = {"-20dB", "0dB", "20dB"};

// --- Main Comparison Application ---
int main()
{
    init_platform();
    printf("\n --- ESPRIT PS vs PL Comparison ---\n");

    // DMA Initialization
    int status;
    XAxiDma_Config *DMA_Config;
    XAxiDma DMA_instance;



    DMA_Config = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    status = XAxiDma_CfgInitialize(&DMA_instance, DMA_Config);

    if (status != XST_SUCCESS)
    {
        printf("DMA Configuration failed.\t\n");
        return 0;
    }

    float ps_times[NUM_SAMPLES];
    float poll_times[NUM_SAMPLES];
    float intr_times[NUM_SAMPLES];
    float ps_total_time = 0;
    float poll_total_time = 0;
    float intr_total_time = 0;

    /* ---- RMSE accumulators ---- */
    int   count_snr[3]        = {0};
    float sum_sq_err_ps_v1[3] = {0.0f};
    float sum_sq_err_ps_v2[3] = {0.0f};
    float sum_sq_err_poll_v1[3] = {0.0f};
    float sum_sq_err_poll_v2[3] = {0.0f};
    float sum_sq_err_intr_v1[3] = {0.0f};
    float sum_sq_err_intr_v2[3] = {0.0f};

    // --- PS LOOP ---
    printf("\n--- Running PS Implementation ---\n");
    printf("%-5s | %-5s | %-25s | %-10s\n", "ID", "SNR", "PS Res", "Time (us)");
    printf("------------------------------------------------------------\n");

    for (int s = 0; s < NUM_SAMPLES; s++) {
        ESPRIT_Sample *sample = &dataset[s];
        XTime ps_start, ps_end;
        float ps_res[2];

        //XTime_SetTime(0);
        XTime_GetTime(&ps_start);
        esprit_ps(sample->data, ps_res);
        XTime_GetTime(&ps_end);

        float ps_time = (float)(ps_end - ps_start) / (COUNTS_PER_SECOND / 1000000);
        ps_times[s] = ps_time;
        ps_total_time += ps_time;

        printf("%-5d | %-5d | (%10.6f, %10.6f) | %10.2f\n",
               sample->sample_id, sample->snr, ps_res[0], ps_res[1], ps_time);

        /* RMSE accumulation for PS */
        int snr_idx;
        switch (sample->snr) {
            case -20: snr_idx = 0; break;
            case   0: snr_idx = 1; break;
            case  20: snr_idx = 2; break;
            default:  snr_idx = -1; break;
        }
        if (snr_idx >= 0) {
            float e_ps1 = sample->gt_vel1 - ps_res[0];
            float e_ps2 = sample->gt_vel2 - ps_res[1];
            sum_sq_err_ps_v1[snr_idx] += e_ps1 * e_ps1;
            sum_sq_err_ps_v2[snr_idx] += e_ps2 * e_ps2;
            count_snr[snr_idx]++;
        }
    }

    float ps_average = ps_total_time / NUM_SAMPLES;
    printf("------------------------------------------------------------\n");
    printf("Average PS Execution Time: %f us\n", ps_average);


    // --- PL LOOP (Polling) ---
    printf("\n--- Running PL Implementation (Polling) ---\n");
    printf("%-5s | %-5s | %-25s | %-10s\n", "ID", "SNR", "PL Res", "Time (us)");
    printf("------------------------------------------------------------\n");
    float pl_input[40];
    float pl_output[2];

    for (int s = 0; s < NUM_SAMPLES; s++) {
        ESPRIT_Sample *sample = &dataset[s];
        XTime poll_start, poll_end;

        // Prepare input buffer
        for(int i=0; i<N_SAMPLES; i++) {
            pl_input[2*i] = sample->data[i].real;
            pl_input[2*i+1] = sample->data[i].imag;
        }

        XTime_GetTime(&poll_start);

        // Send to IP
        status = XAxiDma_SimpleTransfer(&DMA_instance, (UINTPTR)pl_output, 2 * sizeof(float), XAXIDMA_DEVICE_TO_DMA);
        status = XAxiDma_SimpleTransfer(&DMA_instance, (UINTPTR)pl_input, 40 * sizeof(float), XAXIDMA_DMA_TO_DEVICE);

        status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x00000002;
        while (status != 0x00000002)
        {
            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x00000002;
        }

        status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x00000002;                  //0x34 S2MM status register
        while (status != 0x00000002)
        {
            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x00000002;
        }

        XTime_GetTime(&poll_end);

        float poll_time = (float)(poll_end - poll_start) / (COUNTS_PER_SECOND / 1000000);
        poll_times[s] = poll_time;
        poll_total_time += poll_time;

        printf("%-5d | %-5d | (%10.6f, %10.6f) | %10.2f\n",
               sample->sample_id, sample->snr, pl_output[0], pl_output[1], poll_time);

        /* RMSE accumulation for PL Polling */
        int snr_idx;
        switch (sample->snr) {
            case -20: snr_idx = 0; break;
            case   0: snr_idx = 1; break;
            case  20: snr_idx = 2; break;
            default:  snr_idx = -1; break;
        }
        if (snr_idx >= 0) {
            float e_poll1 = sample->gt_vel1 - pl_output[0];
            float e_poll2 = sample->gt_vel2 - pl_output[1];
            sum_sq_err_poll_v1[snr_idx] += e_poll1 * e_poll1;
            sum_sq_err_poll_v2[snr_idx] += e_poll2 * e_poll2;
        }
    }
    float poll_average = poll_total_time / NUM_SAMPLES;
    printf("------------------------------------------------------------\n");
    printf("Average PL (Polling) Execution Time: %f us\n", poll_average);


    // Set up Interrupt system
    status = SetupIntrSystem(&INTCInst, &DMA_instance, XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID, XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID);
    if (status != XST_SUCCESS)
    {
        printf("Failed intr setup\r\n");
        return 0;
    }

    // --- PL LOOP (Interrupts) ---
    printf("\n--- Running PL Implementation (Interrupts) ---\n");
    printf("%-5s | %-5s | %-25s | %-10s\n", "ID", "SNR", "PL Res", "Time (us)");
    printf("------------------------------------------------------------\n");

    for (int s = 0; s < NUM_SAMPLES; s++) {
        ESPRIT_Sample *sample = &dataset[s];
        XTime intr_start, intr_end;

        // Prepare input buffer
        for(int i=0; i<N_SAMPLES; i++) {
            pl_input[2*i] = sample->data[i].real;
            pl_input[2*i+1] = sample->data[i].imag;
        }

        MM2SDone = 0;
        S2MMDone = 0;
        Error = 0;

        XTime_GetTime(&intr_start);

        // Send to IP
        status = XAxiDma_SimpleTransfer(&DMA_instance, (UINTPTR)pl_output, 2 * sizeof(float), XAXIDMA_DEVICE_TO_DMA);
        status = XAxiDma_SimpleTransfer(&DMA_instance, (UINTPTR)pl_input, 40 * sizeof(float), XAXIDMA_DMA_TO_DEVICE);

        while (!(MM2SDone && S2MMDone) && !Error)
        {
        }
        if (Error)
        {
            if (!MM2SDone)
                printf("MM2S is failed\t\n");
            if (!S2MMDone)
                printf("S2MM is failed\t\n");
        }

        XTime_GetTime(&intr_end);

        float intr_time = (float)(intr_end - intr_start) / (COUNTS_PER_SECOND / 1000000);
        intr_times[s] = intr_time;
        intr_total_time += intr_time;

        printf("%-5d | %-5d | (%10.6f, %10.6f) | %10.2f\n",
               sample->sample_id, sample->snr, pl_output[0], pl_output[1], intr_time);

        /* RMSE accumulation for PL Interrupt */
        int snr_idx;
        switch (sample->snr) {
            case -20: snr_idx = 0; break;
            case   0: snr_idx = 1; break;
            case  20: snr_idx = 2; break;
            default:  snr_idx = -1; break;
        }
        if (snr_idx >= 0) {
            float e_intr1 = sample->gt_vel1 - pl_output[0];
            float e_intr2 = sample->gt_vel2 - pl_output[1];
            sum_sq_err_intr_v1[snr_idx] += e_intr1 * e_intr1;
            sum_sq_err_intr_v2[snr_idx] += e_intr2 * e_intr2;
        }
    }
    float intr_average = intr_total_time / NUM_SAMPLES;
    printf("------------------------------------------------------------\n");
    printf("Average PL (Interrupts) Execution Time: %f us\n", intr_average);

    printf("\n --- Overall Comparison ---\n");
    printf("PS Average:            %f us\n", ps_average);
    printf("PL Average (Polling):  %f us\n", poll_average);
    printf("PL Average (Interrupt):%f us\n", intr_average);
    printf("Acceleration Factor (Polling):     %.2f \n", ps_average / poll_average);
    printf("Acceleration Factor (Interrupt):   %.2f \n", ps_average / intr_average);

    /* ------------------------------------------------------------------ */
    /* RMSE summary — combined: sqrt((sum_v1 + sum_v2) / (2 * N))         */
    /* ------------------------------------------------------------------ */
    printf("\n--- RMSE per SNR (combined V1+V2, N=%d samples each) ---\n",
           NUM_SAMPLES / 3);
    printf("%-8s | %-14s | %-19s | %-19s\n", "SNR", "PS RMSE (m/s)", "PL Poll RMSE (m/s)", "PL Intr RMSE (m/s)");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            float rmse_ps = sqrtf((sum_sq_err_ps_v1[i] + sum_sq_err_ps_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            float rmse_poll = sqrtf((sum_sq_err_poll_v1[i] + sum_sq_err_poll_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            float rmse_intr = sqrtf((sum_sq_err_intr_v1[i] + sum_sq_err_intr_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            printf("%-8s | %-14.6f | %-19.6f | %-19.6f\n",
                   snr_label_str[i], rmse_ps, rmse_poll, rmse_intr);
        }
    }
    printf("------------------------------------------------------------------------\n");

    DisconnIntrSystem(&INTCInst, XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID, XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID);

    cleanup_platform();
    return 0;
}
