"""GPU test: gfx12 WMMA multi-row GEMV at rows 2..32 against reconstruct + hgemm.

Loads the model and runs GPU kernels, so run it with nothing else on the GPU. Run it with this checkout's
extension on PYTHONPATH and the runtime environment set (source setup/env.sh, from the recipe); the script forces
EXL3_GEMV_WMMA / _U / _MAX_M per check.

Checks, per projection (attention, GDN, MLP, lm_head; --draft adds the DFlash2 fc and an MLP):
  1. accuracy: rows 2..32 (or --rows), fp16 and fp32 output, U = 1/2/4, against reconstruct_hgemm; the
     upper row tiles (rows 8-15 and 16-31) separately, so a lane-half or M-tile mixup cannot hide
  2. row-tile invariance: row r of an n-row call is bit-identical to row r of the 32-row call, for every
     n in 2..32 (one K order at every row tile M = 8/16/32; this is what keeps a job's logits independent of
     how many other jobs share the verify)
  3. timing: WMMA vs the cooperative GEMM fallback (EXL3_GEMV_WMMA=0) at 8 / 16 / 24 / 32 rows
  4. multi-matrix GEMV (fused gate+up mgemm, the GatedMLP path) at rows 9..32 against reconstruct_hgemm
  5. graphed BC GatedMLP at rows 1..32: per row count 8 calls with distinct inputs (all kept alive), fp32 and
     fp16 output interleaved (one graph per row count and output dtype); every call, eager, capture and
     replay, matches the module's eager path on its own input, and the first matches reconstruct + hgemm
"""
import argparse
import os
import sys
from pathlib import Path

# Python puts this script's directory first on sys.path, ahead of PYTHONPATH; drop it so the checkout
# under test is the one PYTHONPATH names (checked in main)
if sys.path and os.path.abspath(sys.path[0] or os.curdir) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ["EXL3_GEMV_MAX_M"] = "64"

import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import mlp as mlp_mod

ROOT = Path(__file__).resolve().parent.parent
SHAPES = [
    "layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z",
    "layers.0.linear_attn.out_proj", "layers.0.mlp.gate_proj",
    "layers.0.mlp.down_proj", "layers.3.self_attn.k_proj",
    "layers.3.self_attn.q_proj", "lm_head",
]
DRAFT_SHAPES = ["fc", "layers.0.mlp.down_proj", "layers.0.self_attn.q_proj"]


def walk(module):
    yield module
    for child in getattr(module, "modules", []) or []:
        yield from walk(child)


def find_linears(model, suffixes):
    found = {}
    for module in walk(model):
        inner = getattr(module, "inner", None)
        if inner is not None and type(inner).__name__ == "LinearEXL3":
            for suffix in suffixes:
                if module.key == suffix or module.key.endswith("." + suffix):
                    found.setdefault(suffix, inner)
    missing = set(suffixes) - found.keys()
    if missing:
        raise RuntimeError(f"Missing requested projections: {sorted(missing)}")
    return found


def time_call(fn, iterations):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def rel_err(out, ref):
    return ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()


def set_wmma(on, unroll = 4):
    os.environ["EXL3_GEMV_WMMA"] = "1" if on else "0"
    os.environ["EXL3_GEMV_WMMA_U"] = str(unroll)


def check_linears(tag, layers, rows_list, args):
    tested = 0
    print(f"\n[{tag}] 1-3. single-matrix GEMV")
    print("projection rows dtype U rel_err rel_err_rows8-15 rel_err_rows16-31")
    for name, layer in layers.items():
        if not float(layer.K).is_integer():
            print(f"{name}: half-integer bitrate {layer.K}, WMMA not eligible; skipped")
            continue
        tested += 1
        dev = layer.trellis.device
        x32 = torch.randn(1, 64, layer.in_features, dtype = torch.half, device = dev)
        for fp32 in (False, True):
            dtype = torch.float if fp32 else torch.half
            ref32 = layer.reconstruct_hgemm(x32, dtype).float()
            for unroll in (1, 2, 4):
                set_wmma(True, unroll)
                full = layer.bc.run_alloc(x32, layer.out_features, fp32)
                full32 = layer.bc.run_alloc(x32[:, :32].contiguous(), layer.out_features, fp32)
                for rows in rows_list:
                    x = x32[:, :rows].contiguous()
                    out = layer.bc.run_alloc(x, layer.out_features, fp32)
                    o, ref = out.float(), ref32[:, :rows]
                    assert torch.isfinite(o).all(), (name, rows, dtype, unroll)
                    err = rel_err(o, ref)
                    assert err <= args.relative_error, (name, rows, dtype, unroll, err)
                    e8 = rel_err(o[:, 8:16], ref[:, 8:16]) if rows > 8 else float("nan")
                    e16 = rel_err(o[:, 16:], ref[:, 16:]) if rows > 16 else float("nan")
                    for e in (e8, e16):
                        assert not (e == e) or e <= args.relative_error, (name, rows, dtype, unroll, e8, e16)
                    # 2. row-tile invariance (2+ rows: m == 1 takes the fdot2 GEMV, a different K order)
                    if rows >= 2:
                        tile = full32 if rows <= 32 else full
                        same = torch.equal(out, tile[:, :rows])
                        assert same, (f"{name}: rows {rows} differ from the same rows of the {32 if rows <= 32 else 64}-row call "
                                      f"(U={unroll}, {dtype}): max abs diff "
                                      f"{(out.float() - tile[:, :rows].float()).abs().max().item():.3g}")
                    if rows in (2, 8, 9, 16, 17, 24, 32, 33, 48, 64) or args.verbose:
                        print(f"{name} {rows} {dtype} {unroll} {err:.3g} {e8:.3g} {e16:.3g}", flush = True)
        # 3. timing
        for rows in (8, 16, 24, 32, 48, 64):
            x = x32[:, :rows].contiguous()
            set_wmma(False)
            coop = time_call(lambda: layer.bc.run_alloc(x, layer.out_features, False), args.iterations)
            set_wmma(True, 4)
            wmma = time_call(lambda: layer.bc.run_alloc(x, layer.out_features, False), args.iterations)
            print(f"  timing {name} rows {rows}: WMMA {wmma:8.1f} us, fallback (EXL3_GEMV_WMMA=0) {coop:8.1f} us, "
                  f"{coop / wmma:5.2f}x", flush = True)
    assert tested, "no integer-bitrate projections to exercise WMMA"
    return tested


def check_mgemm(model, rows_list, args):
    mlp = next((m for m in walk(model) if type(m).__name__ == "GatedMLP" and m.multi_gu and m.multi_gu[0] is not None),
               None)
    if mlp is None:
        print("\n4. no fused gate+up MultiLinear in this model; multi-matrix check skipped")
        return
    mg, gate, up = mlp.multi_gu[0], mlp.gates[0].inner, mlp.ups[0].inner
    dev = gate.trellis.device
    print(f"\n4. multi-matrix GEMV (fused gate+up, {mlp.key}), rows {rows_list[0]}..{rows_list[-1]}")
    set_wmma(True, 4)
    for rows in rows_list:
        x = torch.randn(1, rows, mlp.hidden_size, dtype = torch.half, device = dev)
        guh = torch.empty((2, rows, mlp.hidden_size), dtype = mlp.interm_dtype, device = dev)
        gu = torch.empty((2, rows, mg.out_features), dtype = mlp.interm_dtype, device = dev)
        ext.exl3_mgemm(x, mg.ptrs_trellis, gu, mg.ptrs_suh, guh, mg.ptrs_svh, None, None, mg.K, -1,
                       mg.mcg, mg.mul1, -1, -1, 0, 1, None, None)
        rg = gate.reconstruct_hgemm(x, mlp.interm_dtype).float()[0]
        ru = up.reconstruct_hgemm(x, mlp.interm_dtype).float()[0]
        eg, eu = rel_err(gu[0].float(), rg), rel_err(gu[1].float(), ru)
        assert eg <= args.relative_error and eu <= args.relative_error, (rows, eg, eu)
        print(f"  rows {rows}: gate {eg:.3g} up {eu:.3g}", flush = True)


def mlp_eager(mlp, x, out_dtype):
    """The module's own eager path (BC graph detached): the same projection kernels, launched one by one
    through Python, so a graph replay must reproduce it"""
    bc = mlp.bc
    mlp.bc = None
    try:
        return mlp.forward(x, {}, out_dtype = out_dtype)
    finally:
        mlp.bc = bc


def check_bc_mlp(model, args):
    mlps = [m for m in walk(model) if type(m).__name__ == "GatedMLP" and m.bc is not None]
    assert mlps, "no GatedMLP with a BC graph"
    mlp = mlps[0]
    limit = getattr(mlp_mod, "_bc_mlp_max_rows", mlp_mod.MAX_BSZN)
    graphs = os.environ.get("EXL3_GRAPHS", "unset")
    print(f"\n5. graphed BC GatedMLP ({mlp.key}), Python row limit {limit}, module out_dtype {mlp.out_dtype}, "
          f"EXL3_GRAPHS {graphs}" + ("" if graphs == "1" else " (ROCm default off: no capture / replay, this checks the "
                                     "direct C++ path; run again with EXL3_GRAPHS=1 for replay)"))
    assert limit >= 32, "this checkout's GatedMLP BC limit is below 32 rows (EXL3_BC_MLP_MAX_ROWS set?)"
    set_wmma(True, 4)
    dev = mlp.device
    g, u, d = mlp.gates[0].inner, mlp.ups[0].inner, mlp.downs[0].inner
    # Per row count: 8 calls alternating fp32 / fp16 output, so each output dtype goes eager, capture,
    # replay, replay, interleaved with the other dtype's graph (a graph keyed by row count alone would
    # replay the other dtype's epilogue into this output). Every call gets its own random input, kept
    # alive, and its own output; all are checked after the last call, each against the eager path on its
    # own input. A replay reading a stale input pointer, or writing a stale / wrongly sized output,
    # lands on another call's data and misses by O(1)
    dtypes = (torch.float, torch.half)
    for rows in list(range(1, 65)):
        xs, outs = [], []
        for call in range(8):
            dt = dtypes[call % 2]
            x = torch.randn(1, rows, mlp.hidden_size, dtype = torch.half, device = dev)
            y = mlp.forward(x, {}, out_dtype = dt)
            assert y.dtype == dt, (rows, call, y.dtype, dt)
            xs.append(x)
            outs.append(y)
        exact = 0
        worst = 0.0
        for call, (x, y) in enumerate(zip(xs, outs)):
            ref = mlp_eager(mlp, x, dtypes[call % 2])
            err = rel_err(y.float(), ref.float())
            worst = max(worst, err)
            exact += int(torch.equal(y, ref))
            assert err <= args.replay_relative_error, \
                f"rows {rows} call {call} ({dtypes[call % 2]}): BC output vs eager on its own input {err:.3g}"
        # Absolute accuracy of the first call: reconstruct + hgemm with the module's activation
        x0 = xs[0]
        gg = g.reconstruct_hgemm(x0, torch.half).float()
        uu = u.reconstruct_hgemm(x0, torch.half).float()
        a = (torch.nn.functional.silu(gg) * uu).half()
        ref0 = d.reconstruct_hgemm(a, torch.float).float()
        err0 = rel_err(outs[0].float(), ref0)
        assert err0 <= args.mlp_relative_error, (rows, err0)
        if rows in (1, 8, 9, 16, 17, 32, 33, 48, 64) or args.verbose:
            print(f"  rows {rows}: 8 calls fp32/fp16 interleaved, max rel_err vs eager {worst:.3g} "
                  f"({exact}/8 bit-identical), rel_err vs reconstruct {err0:.3g}", flush = True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default = str(ROOT / "models/Qwen3.8-27B-EXL3-3.0bpw"))
    parser.add_argument("--draft", nargs = "?", const = str(ROOT / "models/Qwen3.8-27B-DFlash2-EXL3-3.0bpw"),
                        default = None, help = "also check the DFlash2 draft's fc / MLP / q_proj")
    parser.add_argument("--rows", type = int, nargs = "+", default = list(range(1, 65)))
    parser.add_argument("--iterations", type = int, default = 100)
    parser.add_argument("--relative-error", type = float, default = 0.005,
                        help = "max abs error / max reference magnitude")
    parser.add_argument("--mlp-relative-error", type = float, default = 0.02)
    parser.add_argument("--replay-relative-error", type = float, default = 1e-3,
                        help = "BC GatedMLP output vs the module's eager path on the same input")
    parser.add_argument("--verbose", action = "store_true")
    args = parser.parse_args()
    import exllamav3
    pp = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app")]
    loaded = os.path.realpath(exllamav3.__file__)
    assert pp and loaded.startswith(os.path.realpath(pp[0]) + os.sep), \
        f"exllamav3 imported from {loaded}, not from the first PYTHONPATH entry {pp[:1]}"
    print(f"exllamav3 from {os.path.dirname(loaded)}")
    torch.manual_seed(0)

    model = Model.from_config(Config.from_directory(args.model))
    model.load()
    n = check_linears("target", find_linears(model, SHAPES), args.rows, args)
    check_mgemm(model, [r for r in args.rows if r >= 9] or [9, 16, 32], args)
    check_bc_mlp(model, args)
    if args.draft:
        # The draft's lm_head is the target's (attached at generation time), so only its own linears here
        draft = Model.from_config(Config.from_directory(args.draft))
        draft.load()
        n += check_linears("draft", find_linears(draft, DRAFT_SHAPES), args.rows, args)
    print(f"\nPASS: {n} projections, rows {args.rows[0]}..{args.rows[-1]}, fp16/fp32, U=1/2/4; row-tile "
          f"invariance; fused gate+up mgemm; graphed BC GatedMLP rows 1..32")


if __name__ == "__main__":
    main()
