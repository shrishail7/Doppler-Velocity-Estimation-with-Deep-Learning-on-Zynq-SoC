

#include <math.h>
#include "inputs.h"
#include "mlp_run.h"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <xtime_l.h>

#include "platform.h"
#include "xil_printf.h"
#include "xaxidma.h"
#include "xparameters.h"


// ************************************************************ interrupt logic ************************************************************


// step-1

#include "xscugic.h"
#define RESET_TIMEOUT_COUNTER	10000

// step-2
XScuGic INTCInst; //2
/*
 * Flags interrupt handlers use to notify the application context the events.
 */
volatile int MM2SDone;
volatile int S2MMDone;
volatile int Error;

// // step 3
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


// step  4
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



// step 5
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

// step 6
static void DisconnIntrSystem(XScuGic * IntcInstancePtr, u16 MM2SIntrId, u16 S2MMIntrId)
{
	XScuGic_Disconnect(IntcInstancePtr, MM2SIntrId);
	XScuGic_Disconnect(IntcInstancePtr, S2MMIntrId);
}

// ************************************************************ interrupt logic end ************************************************************


const char *snr_labels[3] = {"-20dB", "0dB", "20dB"};

/* Build 40 real inputs from 20 complex samples */
static void build_input(const Complex *data, float *x)
{
    for (int i = 0; i < N_SAMPLES; i++) {
        x[2*i]     = (float)data[i].real;
        x[2*i + 1] = (float)data[i].imag;
    }
}

static void sort2_float(float *a, float *b) {
    if (*a > *b) { float tmp = *a; *a = *b; *b = tmp; }
}

int main(void)
{
    init_platform();
    mlp_init();

    /* Timing */
    XTime PS_Start, PS_End, PL_start, PL_end;
    float total_ps_time = 0.0f;
    float total_pl_time = 0.0f;

    /* RMSE accumulators*/
    int   count_snr[3]        = {0};
    float sum_sq_err_ps_v1[3] = {0.0f};
    float sum_sq_err_ps_v2[3] = {0.0f};
    float sum_sq_err_pl_v1[3] = {0.0f};
    float sum_sq_err_pl_v2[3] = {0.0f};

    /* ------------------------------------------------------------------ */
    /* DMA Initialisation                                                  */
    /* ------------------------------------------------------------------ */
    int status;
    XAxiDma_Config *DMA_Config_ACP;
    XAxiDma        DMA_instance_ACP;

    DMA_Config_ACP = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    status = XAxiDma_CfgInitialize(&DMA_instance_ACP, DMA_Config_ACP);
    if (status != XST_SUCCESS) {
        printf("DMA Configuration Failed.\r\n");
        return 0;
    }

    // ***************************************************** interrupt config *****************************************************
    status = SetupIntrSystem(&INTCInst, &DMA_instance_ACP, XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID, XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID);
        if (status != XST_SUCCESS)
        {
            printf("Failed intr setup\r\n");
            return 0;
        }

        // --- PL LOOP (Interrupts) ---
        printf("\n--- Running PL Implementation (Interrupts) ---\n");
        printf("------------------------------------------------------------\n");



    /* ------------------------------------------------------------------ */
    /* Main inference loop                                                 */
    /* ------------------------------------------------------------------ */
    for (int s = 0; s < NUM_SAMPLES; s++) {

        float x[MLP_N_IN];
        float y[MLP_N_OUT];
        float y_PL[MLP_N_OUT];

        /* ---------- PS inference ---------- */
        XTime_SetTime(0);
        XTime_GetTime(&PS_Start);

        build_input(dataset[s].data, x);
        mlp_forward(x, y);

        XTime_GetTime(&PS_End);
        float local_ps_time = (float)(PS_End - PS_Start) / (COUNTS_PER_SECOND / 1000);
        total_ps_time += local_ps_time;
        // ***************************************************** interrupt config *****************************************************


        MM2SDone = 0;
        S2MMDone = 0;
        Error = 0;


        /* ---------- PL inference via DMA ---------- */
        XTime_SetTime(0);
        XTime_GetTime(&PL_start);

        status = XAxiDma_SimpleTransfer(&DMA_instance_ACP, (UINTPTR)y_PL,
                                        MLP_N_OUT * sizeof(float), XAXIDMA_DEVICE_TO_DMA);
        status = XAxiDma_SimpleTransfer(&DMA_instance_ACP, (UINTPTR)x,
                                        MLP_N_IN  * sizeof(float), XAXIDMA_DMA_TO_DEVICE);

        /* Pooling */
//        status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x00000002;
//        while (status != 0x00000002)
//            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x00000002;
//
//
//        status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x00000002;
//        while (status != 0x00000002)
//            status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x00000002;

        // ***************************************************** interrupt config *****************************************************

        while (!(MM2SDone && S2MMDone) && !Error)
               {
        	if (Error)
        	               {
        	                   if (!MM2SDone)
        	                       printf("MM2S is failed\t\n");
        	                   if (!S2MMDone)
        	                       printf("S2MM is failed\t\n");
        	               }
               }


        XTime_GetTime(&PL_end);
        float local_pl_time = (float)(PL_end - PL_start) / (COUNTS_PER_SECOND / 1000);
        total_pl_time += local_pl_time;

        /* ---------- Mismatch check ---------- */
        for (int i = 0; i < MLP_N_OUT; i++) {
            if (fabsf(y_PL[i] - y[i]) > 0.001f) {
                printf("Mismatch at output %d: ps=%.9g pl=%.9g\r\n", i, y[i], y_PL[i]);
                cleanup_platform();
                return 0;
            }
        }

        /* -------------------------------------------------------------- */
        /* Print: sample number, SNR, then velocity lines                  */
        printf("+ Sample %2d (SNR %4d dB) +\r\n", dataset[s].sample_id, dataset[s].snr);
        for (int i = 0; i < MLP_N_OUT; i++) {
            printf("  for velocity %d\r\n", i + 1);
            printf("  ps = %.9g , pl = %.9g\r\n", y[i], y_PL[i]);
        }
        printf("================================================\r\n\r\n");

        /* ---------- RMSE accumulation (pure float, no double) ---------- */
        int snr_idx;
        switch (dataset[s].snr) {
            case -20: snr_idx = 0; break;
            case   0: snr_idx = 1; break;
            case  20: snr_idx = 2; break;
            default:  snr_idx = -1; break;
        }
        if (snr_idx < 0) continue;

        /* Sort predictions and ground-truth by magnitude before error */
        float sp1 = y[0],    sp2 = y[1];
        float sp1_pl = y_PL[0], sp2_pl = y_PL[1];
        float sg1 = (float)dataset[s].gt_vel1;
        float sg2 = (float)dataset[s].gt_vel2;

        sort2_float(&sp1,    &sp2);
        sort2_float(&sp1_pl, &sp2_pl);
        sort2_float(&sg1,    &sg2);

        float e_ps1 = fabsf(sg1) - fabsf(sp1);
        float e_ps2 = fabsf(sg2) - fabsf(sp2);
        float e_pl1 = fabsf(sg1) - fabsf(sp1_pl);
        float e_pl2 = fabsf(sg2) - fabsf(sp2_pl);

        sum_sq_err_ps_v1[snr_idx] += e_ps1 * e_ps1;
        sum_sq_err_ps_v2[snr_idx] += e_ps2 * e_ps2;
        sum_sq_err_pl_v1[snr_idx] += e_pl1 * e_pl1;
        sum_sq_err_pl_v2[snr_idx] += e_pl2 * e_pl2;
        count_snr[snr_idx]++;
    }

    /* ------------------------------------------------------------------ */
    /* RMSE summary — float sqrtf, printed with full float precision       */
    /* ------------------------------------------------------------------ */
    printf("\r\nRMSE per SNR group (5 samples each):\r\n");
    printf("%-8s | %-20s | %-20s\r\n", "SNR", "PS RMSE (m/s)", "PL RMSE (m/s)");
    printf("---------|---------------------|---------------------\r\n");

    for (int i = 0; i < 3; i++) {
        if (count_snr[i] > 0) {
            float rmse_ps = sqrtf((sum_sq_err_ps_v1[i] + sum_sq_err_ps_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            float rmse_pl = sqrtf((sum_sq_err_pl_v1[i] + sum_sq_err_pl_v2[i])
                                  / (2.0f * (float)count_snr[i]));
            printf("%-8s | %.9g          | %.9g\r\n",
                   snr_labels[i], rmse_ps, rmse_pl);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Timing summary                                                      */
    /* ------------------------------------------------------------------ */
    printf("\r\nTotal execution time PS (ms): %.9g\r\n", total_ps_time);
    printf("Total execution time PL (ms): %.9g\r\n", total_pl_time);
    printf("Average PS (ms): %.9g\r\n", total_ps_time / (float)NUM_SAMPLES);
    printf("Average PL (ms): %.9g\r\n", total_pl_time / (float)NUM_SAMPLES);
    printf("Speed-up ratio: %.9g\r\n",  total_ps_time  / total_pl_time);


    DisconnIntrSystem(&INTCInst, XPAR_FABRIC_AXIDMA_0_MM2S_INTROUT_VEC_ID, XPAR_FABRIC_AXIDMA_0_S2MM_INTROUT_VEC_ID);

    cleanup_platform();
    return 0;
}
