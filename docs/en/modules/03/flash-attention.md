---
outline: [2, 3]
---

# FlashAttention

::: warning Teaching kernel, not the portfolio authority
The `03-hpc-advanced` FlashAttention sources explain tiling and online softmax only. Forward+backward, WMMA, FlashDecoding, and measured numbers live in [cuflash-attn](https://github.com/open-infra-ai/cuflash-attn). The Triton reference is [triton-fused-ops](https://github.com/open-infra-ai/triton-fused-ops).
:::

::: warning Performance numbers are teaching placeholders
The TFLOPS, speedup and memory figures on this page are reference values, not results reproduced by this repository on a pinned hardware/software stack. Re-measure on your own GPU before quoting them.
:::


FlashAttention uses tiling and online softmax to compute attention without materializing the full N×N attention matrix in HBM.

## Standard Attention Problem

```mermaid
sequenceDiagram
    participant HBM as HBM
    participant SRAM as SRAM
    participant Compute as Compute

    HBM ->> SRAM: Load Q, K
    SRAM ->> Compute: S = Q @ K^T
    Compute ->> HBM: Write S (N×N)
    HBM ->> SRAM: Read S
    SRAM ->> Compute: P = softmax(S)
    Compute ->> HBM: Write P (N×N)
    HBM ->> SRAM: Read P, Load V
    SRAM ->> Compute: O = P @ V
    Compute ->> HBM: Write O
```

## FlashAttention Data Flow

```mermaid
sequenceDiagram
    participant HBM as HBM
    participant SRAM as SRAM
    participant Compute as Compute

    HBM ->> SRAM: Load Q_i block
    loop For each KV block j
        HBM ->> SRAM: Load K_j, V_j
        SRAM ->> Compute: S_ij = Q_i @ K_j^T
        Compute ->> SRAM: Online Softmax update
        SRAM ->> Compute: Acc O_i += P_ij @ V_j
    end
    Compute ->> HBM: Write final O_i
```

Total HBM access: O(N) vs O(N²) for standard attention.

## References
