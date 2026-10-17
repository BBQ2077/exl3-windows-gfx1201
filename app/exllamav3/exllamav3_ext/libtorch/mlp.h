#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "linear.h"
#include "../graph.cuh"

#define MAX_BSZN 8   // must match MAX_BSZN in blocksparse_mlp.h / BlockSparseMLP.py

// Rows (bsz * q_len) the graphed dense BC_GatedMLP takes, separate from the MoE kernels' MAX_BSZN:
// a batched speculative verify is bsz x (draft + 1) rows, e.g. 4 x 8. Must match MLP_MAX_ROWS in
// modules/mlp.py, which sizes the static scratch
#define MLP_MAX_ROWS 32

// Gate and up projections run as one fused MGEMM when the pointer tables are given, otherwise
// as two separate GEMV/GEMM calls -- for num_tokens > 1 these go directly through exl3_gemm_gr
// using this class's OWN scratch (down_xh) rather than gate/up/down's BC_LinearEXL3::run_gr,
// which hard-refuses graphing above bsz 1 and whose own xh scratch is a single globally-shared
// cache entry (reused by every quantized Linear layer of the same shape) -- growing that would
// affect the whole model's dense-layer path, not just this one

struct BC_GatedMLP
{
    at::Tensor guh;
    at::Tensor gu;
    at::Tensor a;
    at::Tensor down_xh;
    c10::optional<at::Tensor> gu_ptrs_trellis;
    c10::optional<at::Tensor> gu_ptrs_suh;
    c10::optional<at::Tensor> gu_ptrs_svh;
    float gu_K;
    bool gu_mcg;
    bool gu_mul1;
    bool act_silu;
    bool act_gelu;
    bool act_relu2;
    std::shared_ptr<BC_LinearEXL3> gate;
    std::shared_ptr<BC_LinearEXL3> up;
    std::shared_ptr<BC_LinearEXL3> down;
    float act_limit;

    // graph_bszN[fp32 output][num_tokens - 1] covers num_tokens 1..MLP_MAX_ROWS (num_tokens==1 behaves
    // exactly as the original dedicated bsz-1 path). The output dtype is part of the key: the down
    // projection's epilogue (fp16 or fp32 stores) is baked into the captured kernel, and replaying a
    // graph captured for the other dtype with only the output pointer patched would write the wrong
    // element size into d
    Graph graph_bszN[2][MLP_MAX_ROWS];

    // guh/gu hold TWO slots (gate, up) side by side; the fused mgemm kernel addresses slot j at
    // raw offset j*size_m*size_k from the base pointer, which only lines up with a slice of the
    // (2, MLP_MAX_ROWS, width) static buffer when num_tokens == MLP_MAX_ROWS (slicing dim 1 leaves
    // dim 0's stride at MLP_MAX_ROWS*width, not num_tokens*width, corrupting the slot-1 address for
    // any other num_tokens). Each entry is instead an exactly (2, num_tokens, width)-shaped view of
    // the static's first 2*num_tokens*width elements: contiguous, slot 1 where the kernel expects
    // it, and no per-layer allocation (the statics are shared between layers, like a/down_xh)
    std::vector<at::Tensor> guh_cache;
    std::vector<at::Tensor> gu_cache;

    BC_GatedMLP
    (
        at::Tensor _guh,
        at::Tensor _gu,
        at::Tensor _a,
        at::Tensor _down_xh,
        c10::optional<at::Tensor> _gu_ptrs_trellis,
        c10::optional<at::Tensor> _gu_ptrs_suh,
        c10::optional<at::Tensor> _gu_ptrs_svh,
        float _gu_K,
        bool _gu_mcg,
        bool _gu_mul1,
        bool _act_silu,
        bool _act_gelu,
        bool _act_relu2,
        std::shared_ptr<BC_LinearEXL3> _gate,
        std::shared_ptr<BC_LinearEXL3> _up,
        std::shared_ptr<BC_LinearEXL3> _down,
        float _act_limit
    ) :
        guh                 (std::move(_guh)),
        gu                  (std::move(_gu)),
        a                   (std::move(_a)),
        down_xh             (std::move(_down_xh)),
        gu_ptrs_trellis     (std::move(_gu_ptrs_trellis)),
        gu_ptrs_suh         (std::move(_gu_ptrs_suh)),
        gu_ptrs_svh         (std::move(_gu_ptrs_svh)),
        gu_K                (_gu_K),
        gu_mcg              (_gu_mcg),
        gu_mul1             (_gu_mul1),
        act_silu            (_act_silu),
        act_gelu            (_act_gelu),
        act_relu2           (_act_relu2),
        gate                (_gate),
        up                  (_up),
        down                (_down),
        act_limit           (_act_limit)
    {
        TORCH_CHECK(gu_ptrs_trellis.has_value() || (gate && up), "BC_GatedMLP: need fused mgemm tensors or gate/up handles");
        TORCH_CHECK(guh.dim() == 3 && gu.dim() == 3 && guh.size(0) == 2 && gu.size(0) == 2 &&
                    guh.size(1) >= MLP_MAX_ROWS && gu.size(1) >= MLP_MAX_ROWS &&
                    a.size(1) >= MLP_MAX_ROWS && down_xh.size(1) >= MLP_MAX_ROWS &&
                    guh.is_contiguous() && gu.is_contiguous(),
                    "BC_GatedMLP: scratch must be contiguous (2, MLP_MAX_ROWS, width) / (1, MLP_MAX_ROWS, width)");
        guh_cache.resize(MLP_MAX_ROWS);
        gu_cache.resize(MLP_MAX_ROWS);
    }

    void run_bszN_gr
    (
        const at::Tensor& x,
        at::Tensor& d,
        int num_tokens,
        Graph* graph
    );

    void run_bszN
    (
        const at::Tensor& x,
        at::Tensor& d
    );
};


// Non-gated MLP (up -> act -> down), e.g. NemotronH's relu2 layers. The activation reuses the
// act*y kernels with a ones vector (exact: the multiply is by 1.0f in fp32), applied in place on
// the up output. Padded projection dims stage through zero-padded statics like BC_Attention

struct BC_MLP
{
    c10::optional<at::Tensor> xp;   // (1, K_padded) half, zero-padded input staging
    at::Tensor u;                   // (1, 1, interm) half, up output / activation (in place)
    at::Tensor ones;                // (1, 1, interm) half
    c10::optional<at::Tensor> yp;   // (1, N_padded) out dtype, padded down output
    bool act_silu;
    bool act_gelu;
    bool act_relu2;
    std::shared_ptr<BC_LinearEXL3> up;
    std::shared_ptr<BC_LinearEXL3> down;
    float act_limit;
    int hidden_size;                // exact width of x
    int out_size;                   // exact width of d

    Graph graph_bsz1;

    BC_MLP
    (
        c10::optional<at::Tensor> _xp,
        at::Tensor _u,
        at::Tensor _ones,
        c10::optional<at::Tensor> _yp,
        bool _act_silu,
        bool _act_gelu,
        bool _act_relu2,
        std::shared_ptr<BC_LinearEXL3> _up,
        std::shared_ptr<BC_LinearEXL3> _down,
        float _act_limit,
        int _hidden_size,
        int _out_size
    ) :
        xp          (std::move(_xp)),
        u           (std::move(_u)),
        ones        (std::move(_ones)),
        yp          (std::move(_yp)),
        act_silu    (_act_silu),
        act_gelu    (_act_gelu),
        act_relu2   (_act_relu2),
        up          (_up),
        down        (_down),
        act_limit   (_act_limit),
        hidden_size (_hidden_size),
        out_size    (_out_size)
    {}

    void run_bsz1_gr
    (
        const at::Tensor& x,
        at::Tensor& d,
        Graph* graph
    );

    void run_bsz1
    (
        const at::Tensor& x,
        at::Tensor& d
    );
};