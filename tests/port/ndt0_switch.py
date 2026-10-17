"""DFlash2 草稿數在生成途中切到 0 再切回來：草稿模型要保持同步（切回後每輪 token 數要回到切換前的水準）。
階段：ndt 3 → 0 → 3 → 0 → 3，每階段固定輪數；印每階段每路每輪 token 數與每輪耗時，最後印各路輸出開頭。"""
import os
os.environ.setdefault('EXL3_GEMV_WMMA', '1')
os.environ.setdefault('EXL3_GEMV_MAX_M', '64')
import sys, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "LLM-MODELS-EXL3", "Qwen3.8-27B-EXL3-SC2.20bpw-H3-V3"))
ap.add_argument("--draft", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "LLM-MODELS-EXL3", "Qwen3.8-27B-DFlash2-EXL3-3.0bpw"))
ap.add_argument("--bsz", type=int, default=8)
ap.add_argument("--ndt", type=int, default=3)
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--cs", type=int, default=32768)
args = ap.parse_args()

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
from exllamav3.cache import CacheLayer_quant

kw = dict(layer_type=CacheLayer_quant, k_bits=4, v_bits=4, max_batch_size=args.bsz, max_history=args.ndt)
dcfg = Config.from_directory(args.draft); draft = Model.from_config(dcfg)
dcache = Cache(draft, max_num_tokens=args.cs, **kw); draft.load()
cfg = Config.from_directory(args.model); model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens=args.cs, **kw); model.load()
tok = Tokenizer.from_config(cfg)
gen = Generator(model=model, cache=cache, draft_model=draft, draft_cache=dcache, tokenizer=tok,
                max_batch_size=args.bsz, num_draft_tokens=args.ndt, recurrent_cache_size=512 * 1024**2)

prompts = ["Write a long essay about GPU memory bandwidth and why it limits LLM decoding.",
           "Explain in detail how speculative decoding works, with examples.",
           "Describe the history of the Roman empire in depth.",
           "Write a detailed tutorial about Python asyncio.",
           "List and explain 30 common Linux shell commands with examples.",
           "Write a story about a lighthouse keeper and a storm.",
           "Explain how TCP congestion control works.",
           "Write a C function that parses a CSV line, then explain it line by line."]
def ids(p):
    return tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      encode_special_tokens=True)
jobs = [Job(input_ids=ids(prompts[i % len(prompts)]), max_new_tokens=4000, sampler=ArgmaxSampler(), identifier=i)
        for i in range(args.bsz)]
for j in jobs:
    gen.enqueue(j)
text = {i: "" for i in range(args.bsz)}
def step():
    torch.cuda.synchronize(); t = time.perf_counter()
    for r in gen.iterate():
        if r.get("stage") == "streaming":
            text[r["identifier"]] += r.get("text", "")
    torch.cuda.synchronize(); return time.perf_counter() - t

while not all(j.is_prefill_done() for j in jobs):
    step()
for _ in range(10):
    step()
ok = True
rows = []
for phase, ndt in enumerate([args.ndt, 0, args.ndt, 0, args.ndt]):
    gen.num_draft_tokens = ndt
    before = sum(j.new_tokens for j in jobs)
    dts = [step() for _ in range(args.rounds)]
    got = sum(j.new_tokens for j in jobs) - before
    per = got / args.rounds / args.bsz
    dts.sort()
    rows.append((phase, ndt, per, dts[len(dts) // 2] * 1000, got / sum(dts)))
    print(f"階段 {phase}｜ndt {ndt}｜每路每輪 {per:.2f} token｜每輪中位 {dts[len(dts)//2]*1000:.1f} ms｜總吞吐 {got/sum(dts):.1f} tok/s", flush=True)
draft_phases = [r for r in rows if r[1] > 0]
zero_phases = [r for r in rows if r[1] == 0]
if any(abs(r[2] - 1.0) > 0.01 for r in zero_phases):
    print("FAIL: ndt 0 階段每輪不是剛好 1 token"); ok = False
if min(r[2] for r in draft_phases[1:]) < 0.7 * draft_phases[0][2]:
    print("FAIL: 切回草稿後每輪 token 數掉太多（草稿模型可能失去同步）"); ok = False
for i in range(args.bsz):
    print(f"--- 第 {i} 路（{len(text[i])} 字）:", text[i][:160].replace("\n", " "), "…", text[i][-160:].replace("\n", " "))
print("PASS" if ok else "FAIL")
