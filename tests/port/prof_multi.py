"""2 路 ndt7 每步耗時拆解：用 torch profiler 量 generator.iterate() 每一步，依 kernel／函式排序。"""
import os
os.environ.setdefault('EXL3_GEMV_WMMA', '1')   # 跟 server.py 一致
os.environ.setdefault('EXL3_GEMV_MAX_M', '64')
import os, sys, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "LLM-MODELS-EXL3", "Qwen3.8-27B-EXL3-SC2.20bpw-H3-V3"))
ap.add_argument("--draft", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "LLM-MODELS-EXL3", "Qwen3.8-27B-DFlash2-EXL3-3.0bpw"))
ap.add_argument("--bsz", type=int, default=2)
ap.add_argument("--ndt", type=int, default=7)
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--trace", action="store_true")
ap.add_argument("--cs", type=int, default=16384, help="KV 總容量（server＝路數×每路 Context）")
args = ap.parse_args()

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
from exllamav3.cache import CacheLayer_quant

kw = dict(layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_batch_size=args.bsz, max_history=args.ndt)
if args.ndt > 0:
    dcfg = Config.from_directory(args.draft); draft = Model.from_config(dcfg)
    dcache = Cache(draft, max_num_tokens=args.cs, **kw); draft.load()
else:
    draft = dcache = None   # 不開草稿（num_draft_tokens=0 搭配草稿模型會被當成「用預設長度」）
cfg = Config.from_directory(args.model); model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens=args.cs, **kw); model.load()
tok = Tokenizer.from_config(cfg)
gen = Generator(model=model, cache=cache, draft_model=draft, draft_cache=dcache, tokenizer=tok,
                max_batch_size=args.bsz, num_draft_tokens=args.ndt or None, recurrent_cache_size=512 * 1024**2)

prompts = ["Write a long essay about GPU memory bandwidth and why it limits LLM decoding.",
           "Explain in detail how speculative decoding works, with examples.",
           "Describe the history of the Roman empire in depth.",
           "Write a detailed tutorial about Python asyncio."]
def ids(p):
    return tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      encode_special_tokens=True)
for i in range(args.bsz):
    gen.enqueue(Job(input_ids=ids(prompts[i % len(prompts)]), max_new_tokens=2000, sampler=ArgmaxSampler(), identifier=i))

def step():
    torch.cuda.synchronize(); t = time.perf_counter()
    r = gen.iterate()
    torch.cuda.synchronize(); return time.perf_counter() - t, r

for _ in range(30):   # prefill + warm up
    step()
ts = []
for _ in range(args.steps):
    dt, _ = step(); ts.append(dt)
ts.sort()
import torch as _t
_lh = next(m for m in model.modules if getattr(m, "key", "") == "lm_head")
_in = _lh.inner
print("lm_head:", type(_in).__name__, "is_sliced", _lh.is_sliced, "out", _lh.out_features,
      {k: (str(getattr(_in, k).device), getattr(_in, k).is_pinned() if getattr(_in, k).device.type == "cpu" else "-")
       for k in ("trellis", "weight", "suh", "svh") if isinstance(getattr(_in, k, None), _t.Tensor)},
      "K", getattr(_in, "K", None))
for _r in (8, 16, 32, 64):
    _x = _t.randn(1, _r, 5120, device="cuda:0", dtype=_t.half)
    for _ in range(3): _lh.forward(_x, {})
    _t.cuda.synchronize(); _t0 = time.perf_counter()
    for _ in range(10): _lh.forward(_x, {})
    _t.cuda.synchronize(); print(f"  lm_head 單獨 rows {_r}: {(time.perf_counter()-_t0)/10*1000:.2f} ms")
import torch as _t; _m = _t.cuda.memory_stats(); print(f"cs {args.cs}｜顯存 alloc {_t.cuda.memory_allocated()/2**30:.2f} GiB reserved {_t.cuda.memory_reserved()/2**30:.2f} GiB｜alloc_retries {_m.get('num_alloc_retries')}｜free {_t.cuda.mem_get_info()[0]/2**30:.2f} GiB")
print(f"bsz {args.bsz} ndt {args.ndt}: step median {ts[len(ts)//2]*1000:.1f} ms, min {ts[0]*1000:.1f} ms")


# ---- 依模組類別同步計時 ----
import collections
acc = collections.defaultdict(float); cnt = collections.Counter()
def wrap(m, tag):
    f = m.forward
    def g(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = f(*a, **k)
        torch.cuda.synchronize(); dt_ = time.perf_counter() - t; acc[tag] += dt_; cnt[tag] += 1
        if tag.endswith("Linear"):
            x0 = a[0] if a else None
            dk = f"{tag} {getattr(m, 'key', '?')} x{tuple(x0.shape) if hasattr(x0, 'shape') else '?'}"
            acc["~" + dk] += dt_; cnt["~" + dk] += 1
        return r
    m.forward = g
def walk(mod, prefix, depth):
    for sub in getattr(mod, "modules", []):
        tag = f"{prefix}{type(sub).__name__}"
        if depth == 0 and sub.modules:
            walk(sub, prefix, 1)      # 進到 decoder block 裡
        else:
            wrap(sub, tag)
walk(model, "T.", 0)
if draft is not None: walk(draft, "D.", 0)
for _ in range(3): step()
acc.clear(); cnt.clear()
N = 8; tot = 0.0
for _ in range(N):
    dt, _ = step(); tot += dt
print(f"\n每步 {tot/N*1000:.1f} ms（含同步）")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:40s} {v/N*1000:8.2f} ms/step   calls/step {cnt[k]/N:6.1f}   {v/cnt[k]*1e6:9.1f} us/call")
print(f"  {'（模組以外：取樣、草稿驗證、Python）':40s} {(tot-sum(v for k, v in acc.items() if not k.startswith('~')))/N*1000:8.2f} ms/step")
