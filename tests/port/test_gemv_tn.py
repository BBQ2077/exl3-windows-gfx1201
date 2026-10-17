"""GPU test: EXL3_GEMV_WMMA_TN (kernel 2t, TN adjacent N-tiles per block) against kernel 2w (TN = 1).

Per projection and row count: TN = 2 / 4 output vs reconstruct + hgemm (accuracy) and vs TN = 1 (rounding-level
difference only), row-tile invariance at TN = 2 / 4, then timing TN = 1 / 2 / 4. Also the fused gate+up mgemm
(multi-matrix kernel) at TN = 2 / 4 against reconstruct + hgemm.
Run:  python/python.exe -I -X utf8 -u runtime.py script tests/port/test_gemv_tn.py --model <dir>
"""
import argparse
import os
import sys

if sys.path and os.path.abspath(sys.path[0] or os.curdir) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ["EXL3_GEMV_MAX_M"] = "64"
os.environ["EXL3_GEMV_WMMA"] = "1"
os.environ["EXL3_GEMV_WMMA_U"] = "4"

import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext

SHAPES = [
    "layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z",
    "layers.0.linear_attn.out_proj", "layers.0.mlp.gate_proj",
    "layers.0.mlp.down_proj", "layers.3.self_attn.q_proj", "lm_head",
]


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


def set_tn(tn):
    os.environ["EXL3_GEMV_WMMA_TN"] = str(tn)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True)
    parser.add_argument("--rows", type = int, nargs = "+", default = [2, 8, 9, 16, 17, 24, 32, 33, 48, 64])
    parser.add_argument("--time-rows", type = int, nargs = "+", default = [8, 16, 32, 64])
    parser.add_argument("--iterations", type = int, default = 100)
    parser.add_argument("--relative-error", type = float, default = 0.005)
    args = parser.parse_args()
    torch.manual_seed(0)
    model = Model.from_config(Config.from_directory(args.model))
    model.load()
    layers = find_linears(model, SHAPES)
    total = {1: 0.0, 2: 0.0, 4: 0.0}
    print("projection K rows dtype TN rel_err_vs_ref rel_err_vs_TN1")
    for name, layer in layers.items():
        if not float(layer.K).is_integer() or layer.K > 4:
            print(f"{name}: K {layer.K}, not covered by kernel 2t; skipped")
            continue
        dev = layer.trellis.device
        x64 = torch.randn(1, 64, layer.in_features, dtype = torch.half, device = dev)
        for fp32 in (False, True):
            dtype = torch.float if fp32 else torch.half
            ref64 = layer.reconstruct_hgemm(x64, dtype).float()
            set_tn(1)
            base = {rows: layer.bc.run_alloc(x64[:, :rows].contiguous(), layer.out_features, fp32).float()
                    for rows in args.rows}
            for tn in (2, 4):
                set_tn(tn)
                full32 = layer.bc.run_alloc(x64[:, :32].contiguous(), layer.out_features, fp32)
                full64 = layer.bc.run_alloc(x64, layer.out_features, fp32)
                for rows in args.rows:
                    out = layer.bc.run_alloc(x64[:, :rows].contiguous(), layer.out_features, fp32)
                    o = out.float()
                    assert torch.isfinite(o).all(), (name, rows, dtype, tn)
                    err = rel_err(o, ref64[:, :rows])
                    err1 = rel_err(o, base[rows])
                    assert err <= args.relative_error, (name, rows, dtype, tn, err)
                    assert err1 <= args.relative_error, (name, rows, dtype, tn, err1)
                    if rows >= 2:
                        tile = full32 if rows <= 32 else full64
                        assert torch.equal(out, tile[:, :rows]), (f"{name}: TN {tn} rows {rows} differ from the "
                                                                  f"{32 if rows <= 32 else 64}-row call")
                    if rows in (8, 32, 64):
                        print(f"{name} K{layer.K} {rows} {dtype} {tn} {err:.3g} {err1:.3g}", flush = True)
        for rows in args.time_rows:
            x = x64[:, :rows].contiguous()
            t = {}
            for tn in (1, 2, 4):
                set_tn(tn)
                t[tn] = time_call(lambda: layer.bc.run_alloc(x, layer.out_features, False), args.iterations)
                if name != "lm_head":
                    total[tn] += t[tn] if rows == 32 else 0.0
            print(f"  timing {name} rows {rows}: TN1 {t[1]:7.1f} us  TN2 {t[2]:7.1f} us ({t[1] / t[2]:4.2f}x)  "
                  f"TN4 {t[4]:7.1f} us ({t[1] / t[4]:4.2f}x)", flush = True)

    mlp = next((m for m in walk(model) if type(m).__name__ == "GatedMLP" and m.multi_gu and m.multi_gu[0] is not None),
               None)
    if mlp is not None:
        mg, gate, up = mlp.multi_gu[0], mlp.gates[0].inner, mlp.ups[0].inner
        print(f"\nmulti-matrix (fused gate+up, {mlp.key}, K {mg.K})")
        for rows in (8, 16, 32, 64):
            x = torch.randn(1, rows, mlp.hidden_size, dtype = torch.half, device = gate.trellis.device)
            rg = gate.reconstruct_hgemm(x, mlp.interm_dtype).float()[0]
            ru = up.reconstruct_hgemm(x, mlp.interm_dtype).float()[0]
            t = {}
            for tn in (1, 2, 4):
                set_tn(tn)
                guh = torch.empty((2, rows, mlp.hidden_size), dtype = mlp.interm_dtype, device = x.device)
                gu = torch.empty((2, rows, mg.out_features), dtype = mlp.interm_dtype, device = x.device)
                call = lambda: ext.exl3_mgemm(x, mg.ptrs_trellis, gu, mg.ptrs_suh, guh, mg.ptrs_svh, None, None,
                                              mg.K, -1, mg.mcg, mg.mul1, -1, -1, 0, 1, None, None)
                call()
                eg, eu = rel_err(gu[0].float(), rg), rel_err(gu[1].float(), ru)
                assert eg <= args.relative_error and eu <= args.relative_error, (rows, tn, eg, eu)
                t[tn] = time_call(call, args.iterations)
            print(f"  rows {rows}: gate/up ok; TN1 {t[1]:7.1f} us  TN2 {t[2]:7.1f} us ({t[1] / t[2]:4.2f}x)  "
                  f"TN4 {t[4]:7.1f} us ({t[1] / t[4]:4.2f}x)", flush = True)
    print(f"\nsum of single-matrix timings at 32 rows (excl. lm_head): TN1 {total[1]:.1f} us, TN2 {total[2]:.1f} us, "
          f"TN4 {total[4]:.1f} us")
    print("PASS")


if __name__ == "__main__":
    main()
