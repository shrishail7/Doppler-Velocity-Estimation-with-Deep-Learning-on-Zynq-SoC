# Doppler Velocity Estimation with Deep Learning on Zynq SoC

> **AELD End-Term Project — Group 9**
> Indraprastha Institute of Information Technology, Delhi | Winter Semester 2026
>
> **Team:** Ayush Srivastava (MT25106) · Mansi Jain (MT25125) · Shrishail Dolle (MT25147) · Gagandeep Singh (MT25165)
>
> **Supervisor:** Dr. Sumit Darak | **Mentor:** Aakanksha Tewari

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [System Architecture](#system-architecture)
- [MLP IP Core](#mlp-ip-core)
  - [Network Architecture](#network-architecture)
  - [HLS Source Files](#hls-source-files-mlp)
  - [MLP Pragma Reference](#mlp-pragma-reference)
  - [MLP IP Variants & Results](#mlp-ip-variants--results)
- [ESPRIT IP Core](#esprit-ip-core)
  - [Algorithm Overview](#algorithm-overview)
  - [HLS Source Files](#hls-source-files-esprit)
  - [ESPRIT Pragma Reference](#esprit-pragma-reference)
  - [ESPRIT IP Variants & Results](#esprit-ip-variants--results)
- [Communication Interfaces](#communication-interfaces)
  - [AXI-Stream (DMA)](#axi-stream-dma)
  - [Memory-Mapped (AXI4-Lite)](#memory-mapped-axi4-lite)
  - [Interrupt Controller](#interrupt-controller)
- [RMSE Comparison](#rmse-comparison)
- [Execution Time & Speedup](#execution-time--speedup)
- [Resource Utilisation Summary](#resource-utilisation-summary)
- [Challenges](#challenges)
- [How to Build & Run](#how-to-build--run)
- [Dependencies](#dependencies)

---

## Project Overview

This project implements **real-time Doppler velocity estimation for two simultaneous targets** on a Xilinx Zynq SoC (ZC706 / ZedBoard). Two independent estimation pipelines are accelerated on the Programmable Logic (PL):

| Pipeline | Method | Input | Output |
|---|---|---|---|
| **MLP** | 3-layer Neural Network (trained offline) | 40 real floats (20 complex IQ samples flattened) | 2 velocity estimates |
| **ESPRIT** | Estimation of Signal Parameters via Rotational Invariance Techniques | 20 complex IQ samples (streamed) | 2 velocity estimates |

Both IPs are implemented in **Vitis HLS 2022.1**, exported as AXI-compatible RTL, integrated in **Vivado**, and driven from bare-metal C code running on the **ARM Cortex-A9 PS**.

Four integration strategies are demonstrated for each IP:

| # | Strategy | PS involvement | Data path |
|---|---|---|---|
| 1 | **Unoptimised Streaming** | Polling | AXI-Stream + DMA |
| 2 | **Optimised Streaming** | Polling | AXI-Stream + DMA |
| 3 | **Memory-Mapped** | Polling | AXI4 Master (memcpy) |
| 4 | **Interrupt Controller** | ISR-driven | AXI-Stream + DMA + GIC |

---

## Repository Structure

```
.
├── MLP/
│   ├── HLS/
│   │   ├── main.h               # Top-level interface header
│   │   ├── main.cpp             # HLS top function (AXI-Stream wrapper)
│   │   ├── mlp_run.h            # MLP dimensions & function declarations
│   │   ├── mlp_run.cpp          # HLS inference engine (with pragmas)
│   │   ├── mlp_weights_data.h   # Trained float weights (fc1/fc2/fc3, LN params)
│   │   ├── inputs.h             # 15-sample test dataset
│   │   └── main_tb.cpp          # C-simulation testbench
│   └── IDE/
│       └── main.c               # Bare-metal ARM application (DMA + timing + RMSE)
│
├── ESPRIT/
│   ├── HLS/
│   │   ├── esprit.h             # Top-level interface header
│   │   ├── esprit.cpp           # Full ESPRIT pipeline (HLS)
│   │   └── esprit_tb.cpp        # C-simulation testbench
│   └── IDE/
│       └── helloworld.c         # Bare-metal ARM application
│
└── README.md
```

---

## System Architecture

```
┌──────────────────────── Zynq SoC ─────────────────────────────────┐
│                                                                     │
│  ┌──────────────────────────────┐   ┌──────────────────────────┐  │
│  │   ARM Cortex-A9 (PS)         │   │   FPGA Fabric (PL)       │  │
│  │                              │   │                           │  │
│  │  bare-metal C application    │   │  ┌────────────┐           │  │
│  │  ┌──────────────────────┐    │   │  │  MLP IP    │           │  │
│  │  │ build_input()        │◄───┼───┼──┤  (HLS)     │           │  │
│  │  │ mlp_forward() [SW]   │    │   │  └────────────┘           │  │
│  │  │ DMA SimpleTransfer() │────┼───┼──►AXI-DMA                 │  │
│  │  │ XTime_GetTime()      │    │   │  ┌────────────┐           │  │
│  │  │ RMSE accumulators    │    │   │  │ ESPRIT IP  │           │  │
│  │  └──────────────────────┘    │   │  │  (HLS)     │           │  │
│  │                              │   │  └────────────┘           │  │
│  └──────────────────────────────┘   └──────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Data flow (DMA variant):
```
PS writes x[40] → DMA MM2S → MLP/ESPRIT IP → DMA S2MM → PS reads y[2]
```

---

## MLP IP Core

### Network Architecture

```
Input (40)
    │
    ▼
 FC1 (40→400) ──► LayerNorm ──► ReLU   [h1: 400]
    │
    ▼
 FC2 (400→400) ──► LayerNorm ──► ReLU  [h2: 400]
    │
    ▼
 FC3 (400→2)                            [y:  2]
```

- **Input:** 20 complex IQ samples flattened to 40 real floats
- **Weights:** stored as C `const float[]` arrays in `mlp_weights_data.h`
- **Activation:** ReLU after each LayerNorm
- **Output:** two Doppler velocity estimates (float)

### HLS Source Files (MLP)

#### `main.h` — Top-level interface

```cpp
typedef float Data_type;
typedef hls::axis<Data_type, 0, 0, 0> axis_data;

#define Input_Size  40
#define Output_Size  2

void MLP(hls::stream<axis_data> &Input, hls::stream<axis_data> &Output);
```

#### `main.cpp` — AXI-Stream top function

```cpp
void MLP(hls::stream<axis_data> &Input, hls::stream<axis_data> &Output)
{
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=Input
#pragma HLS INTERFACE axis register both port=Output

    mlp_init();                          // print banner (SW only)

    Data_type InputA[Input_Size];
#pragma HLS ARRAY_PARTITION variable=InputA type=complete dim=1

    Data_type OutputY[Output_Size];
#pragma HLS ARRAY_PARTITION variable=OutputY type=complete dim=1

    axis_data pkt_in, pkt_out;

    READ_IN:
    for (int i = 0; i < Input_Size; ++i) {
#pragma HLS PIPELINE II=1
        pkt_in    = Input.read();
        InputA[i] = pkt_in.data;
    }

    mlp_forward(InputA, OutputY);        // inference

    WRITE_OUT:
    for (int i = 0; i < Output_Size; ++i) {
#pragma HLS PIPELINE II=1
        pkt_out.data = OutputY[i];
        pkt_out.keep = -1;  pkt_out.strb = -1;
        pkt_out.last = (i == Output_Size - 1) ? 1 : 0;
        Output.write(pkt_out);
    }
}
```

#### `mlp_run.cpp` — Inference engine

```cpp
#define LN_EPS   1e-5f
#define FADD_LAT 5          // float adder pipeline depth

static inline float relu_f(float v) { return v > 0.0f ? v : 0.0f; }

// Fully-connected layer
static inline void linear_layer(
        const float *in,  int n_in,
        float       *out, int n_out,
        const float *W,   const float *b)
{
#pragma HLS INLINE
    OUT_LOOP: for (int o = 0; o < n_out; ++o) {
#pragma HLS PIPELINE II=1
        float acc = b[o];
        const float *row = W + o * n_in;
        MAC_LOOP: for (int i = 0; i < n_in; ++i) {
#pragma HLS UNROLL factor=40
            acc += row[i] * in[i];
        }
        out[o] = acc;
    }
}

// Layer normalisation + ReLU (in-place safe)
static inline void layer_norm_relu(
        const float *in, float *out, int n,
        const float *gamma, const float *beta)
{
#pragma HLS INLINE
    // -- (1) mean  (5-way partial-sum to hide FADD latency) --
    float sum_p[FADD_LAT];
#pragma HLS ARRAY_PARTITION variable=sum_p complete dim=1
    for (int k = 0; k < FADD_LAT; ++k) sum_p[k] = 0.0f;
    MEAN_LOOP: for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        sum_p[i % FADD_LAT] += in[i];
    }
    float mean = (sum_p[0]+sum_p[1]+sum_p[2]+sum_p[3]+sum_p[4]) / (float)n;

    // -- (2) variance --
    float var_p[FADD_LAT];
#pragma HLS ARRAY_PARTITION variable=var_p complete dim=1
    for (int k = 0; k < FADD_LAT; ++k) var_p[k] = 0.0f;
    VAR_LOOP: for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        float d = in[i] - mean;
        var_p[i % FADD_LAT] += d * d;
    }
    float var = (var_p[0]+var_p[1]+var_p[2]+var_p[3]+var_p[4]) / (float)n;
    float inv_std = 1.0f / sqrtf(var + LN_EPS);

    // -- (3) normalise + scale + ReLU --
    NORM_LOOP: for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        float norm   = (in[i] - mean) * inv_std;
        float scaled = gamma[i] * norm + beta[i];
        out[i]       = relu_f(scaled);
    }
}
```

---

### MLP Pragma Reference

| Pragma | Location | Purpose |
|--------|----------|---------|
| `#pragma HLS INTERFACE ap_ctrl_none port=return` | `MLP()` top | Removes handshake ports; IP starts immediately on valid AXI-Stream data |
| `#pragma HLS INTERFACE axis register both port=Input/Output` | `MLP()` top | Exposes AXI4-Stream slave/master ports with registered handshake signals |
| `#pragma HLS ARRAY_PARTITION variable=InputA type=complete dim=1` | `MLP()` | Splits entire 40-element input array into individual registers — removes BRAM port bottleneck, enables single-cycle random access |
| `#pragma HLS ARRAY_PARTITION variable=OutputY type=complete dim=1` | `MLP()` | Same as above for 2-element output array |
| `#pragma HLS PIPELINE II=1` on `READ_IN` / `WRITE_OUT` | `MLP()` | One AXI-Stream beat per clock cycle for both ingress and egress loops |
| `#pragma HLS INLINE` | `linear_layer()`, `layer_norm_relu()` | Merges helper functions into `mlp_forward` at synthesis time — exposes all internal resources to the scheduler for cross-boundary optimisation |
| `#pragma HLS PIPELINE II=1` on `OUT_LOOP` | `linear_layer()` | Forces the output-neuron loop to accept a new iteration every clock — combined with UNROLL below this achieves near-peak MAC throughput |
| `#pragma HLS UNROLL factor=40` on `MAC_LOOP` | `linear_layer()` | Creates 40 parallel multiplier-adder units; one complete dot-product row is reduced in a tree each outer cycle (matches input width of 40) |
| `#pragma HLS PIPELINE II=1` on `MEAN_LOOP`, `VAR_LOOP`, `NORM_LOOP` | `layer_norm_relu()` | Starts a new element every cycle in all three LN passes |
| `#pragma HLS ARRAY_PARTITION variable=sum_p/var_p complete dim=1` | `layer_norm_relu()` | Exposes all 5 partial-sum accumulators as independent registers, breaking the loop-carried dependency chain that would otherwise stall the pipeline |
| `#pragma HLS BIND_STORAGE variable=fc1_W/fc2_W/fc3_W type=ROM_2P impl=BRAM` | `mlp_forward()` | Instructs HLS to map weight arrays to dual-port Block RAM — two reads per cycle supported |
| `#pragma HLS ARRAY_PARTITION variable=fc1_W/fc2_W/fc3_W type=cyclic factor=40 dim=1` | `mlp_forward()` | Splits each weight BRAM into 40 cyclic banks (bank *k* holds indices *k, k+40, k+80, …*) — provides 40 simultaneous read ports matching the UNROLL factor |
| `#pragma HLS ARRAY_PARTITION variable=fc1_b/fc2_b/fc3_b type=complete dim=1` | `mlp_forward()` | Bias arrays fully partitioned into registers for zero-latency access |
| `#pragma HLS ARRAY_PARTITION variable=ln1_g/ln1_b/ln2_g/ln2_b type=cyclic factor=40 dim=1` | `mlp_forward()` | LayerNorm gamma/beta partitioned to match parallel NORM_LOOP access |
| `#pragma HLS ARRAY_PARTITION variable=h1/h2 type=cyclic factor=40 dim=1` | `mlp_forward()` | Hidden-layer buffers split into 40 banks so FC2/FC3 input reads are conflict-free |

---

### MLP IP Variants & Results

#### C-Simulation (HLS testbench)

All 15 samples passed with Benchmark == IP stream output for all SNR groups.

#### Synthesis Results

| IP Variant | Latency (cycles) | BRAM | DSP | FF | LUT |
|---|---|---|---|---|---|
| **Unoptimised** | 662,831 | 558 (51%) | 10 (1%) | 8,417 (1%) | 7,698 (3%) |
| **Optimised** | 278,800 | 564 (37%) | 18 (~0%) | 13,288 (2%) | 14,920 (5%) |
| **Memory-Mapped** | 176,398 | 365 (33%) | 18 (2%) | 26,890 (6%) | 32,982 (15%) |
| **Aggressively Opt.** | 11,697 | 400 (36%) | 400 (44%) | 182,206 (41%) | 138,558 (63%) |

> **Board:** ZC706 (xc7z045-ffg900-1) — all results target 10 ns clock period.

---

## ESPRIT IP Core

### Algorithm Overview

ESPRIT estimates frequencies (mapped to Doppler velocities) from a covariance matrix of received complex samples:

```
IQ samples (N=20)
       │
       ▼
  Spatial Smoothing  ──►  Autocorrelation matrix R [N×N]
       │
       ▼
  QR Eigendecomposition  (Jacobi/Givens iterations)
       │
       ▼
  Subspace Extraction  ──►  subA, subB  [L×k]
       │
       ▼
  SVD pseudo-inverse  ──►  pinvA
       │
       ▼
  Phi matrix  ──►  Eigenvalues of Phi
       │
       ▼
  atan2 → angle → velocity  (2 estimates, sorted & clipped)
```

### HLS Source Files (ESPRIT)

#### `esprit.h` — Top-level interface

```cpp
#define N_SAMPLES 20
typedef float ESPRIT_Dtype;

typedef struct { float real; float imag; } Complex;
typedef hls::axis<ESPRIT_Dtype, 0, 0, 0> axis_data;

#define ESPRIT_IP
#ifdef ESPRIT_IP
void esprit_hls(hls::stream<axis_data> &in_stream,
                hls::stream<axis_data> &out_stream);
#endif
```

#### Key functions in `esprit.cpp`

| Function | Description |
|---|---|
| `fast_atan2f()` | Software-rational atan2 approximation (avoids FP divide chain) |
| `conj_complex()` | Complex conjugate |
| `matmul_complex()` | Generic *n×n* complex matrix multiply |
| `transpose_conj_matrix()` | Hermitian transpose |
| `spatial_smoothing()` | Builds smoothed autocorrelation matrix from received signal |
| `qr_givens_S()` | QR decomposition via Givens rotations |
| `qr_algorithm_eig_single()` | Iterative QR eigenvalue algorithm |
| `sort_eigenvalues()` | Bubble sort eigenvalues descending |
| `svd_pinv_complex_k2()` | SVD-based pseudo-inverse (k=2 targets) |
| `eign_cal_ES()` | Eigenvalue calculation of 2×2 matrix (closed form) |
| `vel_clip()` | Clips output to [-30, +30] m/s physical range |
| `esprit_hls()` | HLS top: read stream → run pipeline → write stream |

---

### ESPRIT Pragma Reference

| Pragma | Location | Purpose |
|--------|----------|---------|
| `#pragma HLS INTERFACE ap_ctrl_none port=return` | `esprit_hls()` | Free-running IP; no external start/done handshake needed |
| `#pragma HLS INTERFACE axis register both port=in_stream/out_stream` | `esprit_hls()` | AXI4-Stream slave (input) and master (output) with registered ready/valid |
| `#pragma HLS PIPELINE II=1` on `READ_L1` | `esprit_hls()` | One complex sample ingested per clock (two beats: real then imag) |
| `#pragma HLS INLINE` | All helper math functions | Merges the entire algorithm into one scheduling region so the tool can overlap and pipeline across function boundaries |
| `#pragma HLS PIPELINE II=1` on `MATMUL_I` / `MATMUL_J` | `matmul_complex()` | Inner loops of matrix multiply start a new iteration every cycle |
| `#pragma HLS UNROLL` on inner matrix loops | `matmul_complex()`, `copy_matrix()`, `identity_matrix()` | Full unroll of small inner loops — eliminates loop overhead for fixed small dimensions |
| `#pragma HLS PIPELINE II=1` on `TRANSPOSE_J` | `transpose_conj_matrix()` | Column iteration fully pipelined |
| `#pragma HLS PIPELINE II=1` on `SS_CLEAR` | `spatial_smoothing()` | Zeros accumulator arrays in one cycle per element |
| `#pragma HLS PIPELINE II=1` on `SS_ROW` / `SS_COL` | `spatial_smoothing()` | Inner correlation accumulation pipelined for maximum throughput |
| `#pragma HLS PIPELINE II=1` on `QR_COL` | `qr_givens_S()` | QR Givens rotation update loop pipelined |
| `#pragma HLS PIPELINE II=1` on `QR_UPDATE` | `qr_givens_S()` | Matrix update after each Givens step pipelined |
| `#pragma HLS PIPELINE II=1` on `SORT_SWAP` | `sort_eigenvalues()` | Swap loop pipelined |
| `#pragma HLS PIPELINE II=1` on `EXTRACT_L1/L2` | `esprit_hls()` | Subspace extraction from eigenvector matrix pipelined |
| `#pragma HLS PIPELINE II=1` on `PHI_L1/L2/L3` | `esprit_hls()` | Phi matrix construction loops pipelined |
| `#pragma HLS PIPELINE II=1` on `CALC_L1` | `esprit_hls()` | atan2 angle computation pipelined |
| `#pragma HLS ARRAY_PARTITION variable=A/Q/R/TempA complete dim=2` | `qr_algorithm_eig_single()` (Optimised) | Provides multiple read/write ports to local matrices, resolving resource-contention bottlenecks in the QR inner loops |
| `#pragma HLS ARRAY_PARTITION variable=eigenvectors/eigenvalues/TempQTotal complete dim=2` | `qr_algorithm_eig_single()` (Optimised) | Same — full column access in one cycle for eigenvector updates |

#### Memory-Mapped ESPRIT variant additional pragmas

```cpp
// esprit.h (MM variant)
void esprit_hls(Complex *in_mem, float *out_mem);

// esprit.cpp
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi port=in_mem  offset=slave depth=20
#pragma HLS INTERFACE m_axi port=out_mem offset=slave depth=2
#pragma HLS ARRAY_PARTITION variable=rec_signal type=complete dim=0
```

| Extra Pragma | Purpose |
|---|---|
| `INTERFACE s_axilite port=return` | Exposes AXI4-Lite control register (start/done/idle) |
| `INTERFACE m_axi port=in_mem/out_mem` | IP acts as AXI4 master — bursts data directly from/to DDR without CPU involvement |
| `ARRAY_PARTITION … complete dim=0` | Fully partitions the local `rec_signal` copy — eliminates all BRAM bottlenecks in subsequent computations |

---

### ESPRIT IP Variants & Results

#### Synthesis Results

| IP Variant | Board | Latency (cycles) | BRAM | DSP | FF | LUT |
|---|---|---|---|---|---|---|
| **Unoptimised** | ZedBoard | 221,931 | 34 (12%) | 72 (52%) | 23,957 (22%) | 32,823 (61%) |
| **Optimised** | ZC706 | 62,479 | 2 (0%) | 658 (73%) | 231,326 (52%) | 204,573 (93%) |
| **Memory-Mapped** | ZC706 | 73,604 | 10 (0%) | 686 (76%) | 212,522 (48%) | 194,398 (88%) |

> The optimised variant reduced latency from ~220k cycles to ~62k cycles — a **3.5× improvement** — primarily by unrolling inner matrix loops and partitioning intermediate matrices.

---

## Communication Interfaces

### AXI-Stream (DMA)

```
PS (ARM)                    PL (FPGA)
────────                    ─────────
XAxiDma_SimpleTransfer ──MM2S──► IP input stream
                       ◄─S2MM── IP output stream
XAxiDma_SimpleTransfer

// Polling variant
while ((XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x04) & 0x2) != 0x2);
while ((XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR, 0x34) & 0x2) != 0x2);
```

### Memory-Mapped (AXI4-Lite)

```cpp
// MLP MM variant — IP performs its own DDR burst
#pragma HLS INTERFACE m_axi port=Input  depth=40 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=Output depth=2  offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=Input  bundle=control
#pragma HLS INTERFACE s_axilite port=Output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

mlp_init();
memcpy(InputA, (const Data_type*)Input,  Input_Size  * sizeof(Data_type));
mlp_forward(InputA, OutputY);
memcpy((Data_type*)Output, OutputY, Output_Size * sizeof(Data_type));
```

### Interrupt Controller

```c
// ISR for MM2S completion
static void MM2SIntrHandler(void *Callback) {
    u32 IrqStatus = XAxiDma_IntrGetIrq(AxiDmaInst, XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_IntrAckIrq(AxiDmaInst, IrqStatus, XAXIDMA_DMA_TO_DEVICE);
    if (IrqStatus & XAXIDMA_IRQ_IOC_MASK) MM2SDone = 1;
}

// Main loop waits on flags rather than busy-polling registers
while (!(MM2SDone && S2MMDone) && !Error) { /* CPU free for other tasks */ }
```

**GIC Setup (key steps):**
1. `XScuGic_LookupConfig` + `XScuGic_CfgInitialize`
2. `Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT, XScuGic_InterruptHandler, …)`
3. `XScuGic_Connect` for both MM2S and S2MM interrupt IDs
4. `XAxiDma_IntrEnable(XAXIDMA_IRQ_ALL_MASK, …)` on both channels
5. `Xil_ExceptionEnable()`

---

## RMSE Comparison

### MLP (all four IP variants are numerically identical)

| SNR | PS RMSE (m/s) | PL RMSE (m/s) |
|---|---|---|
| −20 dB | 4.9055 | 4.9055 |
| 0 dB | 2.9650 | 2.9650 |
| +20 dB | 1.7343 | 1.7343 |

> PS and PL outputs match to within 1×10⁻³ m/s for all 15 test samples across all variants.

### ESPRIT (all four IP variants)

| SNR | PS RMSE (m/s) | PL RMSE (m/s) |
|---|---|---|
| −20 dB | 17.51 | 17.51 |
| 0 dB | 10.38 | 10.38 |
| +20 dB | 5.23 | 5.23 |

> ESPRIT RMSE is inherently higher than MLP at low SNR — the neural network generalises better under noise. Both improve monotonically with SNR.

---

## Execution Time & Speedup

### MLP (ZC706, 15 samples)

| IP Configuration | Avg PL Time (ms) | Speedup vs Unoptimised |
|---|---|---|
| Unoptimised | 13.248 | 1× |
| Optimised | 5.551 | **2.39×** |
| Memory-Mapped | 3.498 | **3.79×** |
| Interrupt Controller | 5.552 | **2.39×** |

> PS (software-only) average: ~2.16 ms — PL accelerated variants are slower for this small model due to DMA transfer overhead. Memory-mapped provides the best throughput.

### ESPRIT (ZC706, 15 samples)

| IP Configuration | Avg PL Time (ms) | Speedup vs Unoptimised |
|---|---|---|
| Unoptimised | 1.450 | 1× |
| Optimised (Stream) | 0.763 | **1.90×** |
| Memory-Mapped | 0.927 | **1.56×** |
| Interrupt Controller | 0.878 | **1.65×** |

> PS average: ~5.9 ms → **overall PL speedup 7.74×** for the optimised streaming variant.

---

## Resource Utilisation Summary

| IP | Variant | BRAM | DSP | FF | LUT |
|---|---|---|---|---|---|
| MLP | Unoptimised | 558 (51%) | 10 (1%) | 8,417 (1%) | 7,698 (3%) |
| MLP | Optimised | 564 (37%) | 18 (~0%) | 13,288 (2%) | 14,920 (5%) |
| MLP | Memory-Mapped | 365 (33%) | 18 (2%) | 26,890 (6%) | 32,982 (15%) |
| ESPRIT | Unoptimised | 34 (12%) | 72 (52%) | 23,957 (22%) | 32,823 (61%) |
| ESPRIT | Optimised | 2 (0%) | 658 (73%) | 231,326 (52%) | 204,573 (93%) |
| ESPRIT | Memory-Mapped | 10 (0%) | 686 (76%) | 212,522 (48%) | 194,398 (88%) |

---

## Challenges

### 1. ZedBoard Resource Constraint
The MLP network with 400-neuron hidden layers exceeded the available BRAM and LUT budget on the ZedBoard (xc7z020). The design was migrated to the ZC706 (xc7z045) to accommodate the aggressive array-partitioning optimisations.

### 2. Fixed-Point Synthesis Timeout
An attempt to use `ap_fixed<21,3>` (3 integer bits, 18 fractional bits) to reduce resource usage required synthesising ~200,000 float-to-fixed conversions. Synthesis exceeded **20 hours** and terminated with a timeout.

### 3. Out-of-Memory During Implementation
The aggressively optimised MLP variant (cyclic factor=40 on weight BRAMs + full partitioning of all biases) reduced latency from 662k to 11.7k cycles but caused Vivado's `generate_output_products` to consume **>8 GB RAM over 30+ hours** before failing with an OOM error, illustrating the area–performance trade-off.

### 4. ESPRIT LUT Pressure
The optimised ESPRIT IP consumes 93% of ZC706 LUTs, leaving minimal margin. The timing violation (`II&Timing Violation` reported in HLS) required careful pipeline-depth tuning to converge at the 10 ns target clock.

---

## How to Build & Run

### HLS (Vitis HLS 2022.1)

```bash
# MLP
vitis_hls -f run_hls.tcl   # synthesises main.cpp + mlp_run.cpp
# C-Simulation
vitis_hls -p MLP_project    # → Run C Simulation (main_tb.cpp)
```

**Project settings:**
- Top function: `MLP`
- Target device: `xc7z045ffg900-2` (ZC706)
- Clock period: 10 ns

```bash
# ESPRIT
vitis_hls -f run_hls.tcl   # synthesises esprit.cpp
# C-Simulation testbench: esprit_tb.cpp
```

### Vivado Block Design

1. Import generated HLS IP (`solution1/impl/ip`)
2. Add **AXI DMA** (Simple mode, 32-bit, 4 KB burst)
3. Connect `M_AXIS_MM2S → IP input stream`, `IP output stream → S_AXIS_S2MM`
4. Add **AXI Interconnect** and **Processing System 7** (ZC706 preset)
5. Run Connection Automation → Validate → Generate Bitstream

### Vitis IDE (Bare-metal)

1. Create platform from XSA exported by Vivado
2. Add `main.c` (or `helloworld.c` for ESPRIT) as application source
3. Include `inputs.h` and `mlp_run.h/mlp_run.c` in the project
4. Build → Program FPGA → Run (JTAG UART terminal at 115200 baud)

---

## Dependencies

| Tool | Version |
|---|---|
| Vitis HLS | 2022.1 |
| Vivado | 2022.1 |
| Vitis IDE | 2022.1 |
| Target board (MLP) | ZC706 (xc7z045ffg900-2) |
| Target board (ESPRIT Unopt.) | ZedBoard (xc7z020clg484-1) |
| Target board (ESPRIT Opt.) | ZC706 (xc7z045ffg900-2) |
| Standalone BSP | Xilinx Standalone v8.x |
| AXI DMA driver | `xaxidma.h` (Xilinx SDK) |
| GIC driver | `xscugic.h` (Xilinx SDK) |

---

## Work Distribution

| Member | Contributions |
|---|---|
| **Gagandeep Singh** | ESPRIT Unoptimised IP · ESPRIT Interrupt-Controller IP · Block diagrams · PS vs PL verification |
| **Shrishail Dolle** | MLP Unoptimised IP · MLP Interrupt-Controller IP · Block diagrams · PS vs PL verification |
| **Mansi Jain** | ESPRIT Optimised IP · ESPRIT Memory-Mapped IP · Block diagrams · PS vs PL verification |
| **Ayush Srivastava** | MLP Optimised IP · MLP Memory-Mapped IP · Block diagrams · PS vs PL verification |

---

*Report and implementation — Group 9, IIIT Delhi, Winter 2026*