"""單層 GatedMLP／GDN／Attention 在 8 vs 16 列的耗時拆解。"""
import os, sys, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "LLM-MODELS-EXL3", "Qwen3.8-27B-EXL3-SC2.20bpw-H3-V3"))
ap.add_argument("--layer", type=int, default=1)
args = ap.parse_args()
import torch
from exllamav3 import Config, Model
cfg = Config.from_directory(args.model); model = Model.from_config(cfg); model.load()

def find(cls_name, idx):
    hits = []
    for m in model:
        if type(m).__name__ == cls_name: hits.append(m)
    return hits[idx]

def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n * 1000  # us

mlp = find("GatedMLP", args.layer)
print("MLP:", mlp.key, "gate", type(mlp.gates[0]).__name__ if hasattr(mlp, "gates") else "", flush=True)
print("bc:", mlp.bc is not None, "multi_gu:", [x is not None for x in getattr(mlp, "multi_gu", [])])
H = cfg.hidden_size
for rows in (1, 4, 8, 9, 12, 16, 24, 32):
    x = torch.randn(1, rows, H, device="cuda", dtype=torch.half) * 0.1
    t_mlp = bench(lambda: mlp.forward(x, {}))
    xs = x.view(rows, H)
    g, u, d = mlp.gates[0], mlp.ups[0], mlp.downs[0]
    t_g = bench(lambda: g.forward(x, {}))
    t_u = bench(lambda: u.forward(x, {}))
    xi = torch.randn(1, rows, d.in_features, device="cuda", dtype=torch.half) * 0.1
    t_d = bench(lambda: d.forward(xi, {}))
    print(f"rows {rows:2d}: MLP {t_mlp:8.1f} us | gate {t_g:7.1f} up {t_u:7.1f} down {t_d:7.1f} us  (sum {t_g+t_u+t_d:7.1f})", flush=True)
