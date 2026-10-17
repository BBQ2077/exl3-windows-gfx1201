"""GPU test: greedy DFlash2 output of one checkout, saved for an exact comparison against another.

Loads the target + DFlash2 draft like TabbyAPI (Q4 caches, max_history = draft tokens), runs each prompt
greedily, and writes the generated token ids to --out. Run it once per checkout (PYTHONPATH selects the tree),
then --compare the two files: every prompt must match token for token. exllamav3 must import from the first
PYTHONPATH entry (checked; this script's own directory is dropped from sys.path), and --compare requires
both files to hold this script's full prompt set.

  source setup/env.sh (from the recipe)
  PYTHONPATH=/path/to/exllamav3-baseline python test_greedy_equal.py --out /tmp/greedy_baseline.json
  PYTHONPATH=/path/to/exllamav3-rdna4    python test_greedy_equal.py --out /tmp/greedy_rdna4.json
  python test_greedy_equal.py --compare /tmp/greedy_baseline.json /tmp/greedy_rdna4.json

--bsz N runs the prompts as N concurrent jobs instead of one at a time. A batched verify is not expected to
be bit-identical to bs1 (the split-decode attention splits differently at another batch size, so its combine
rounds differently), so compare a batched file against a bs1 file for the common-prefix lengths only.
"""
import argparse, json, os, sys, time

# Python puts this script's directory (the checkout it ships in) first on sys.path, ahead of PYTHONPATH, so
# without this every run would import that checkout whatever PYTHONPATH says. Drop that entry only: when
# PYTHONPATH names this same checkout, its own (later) entry stays
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or os.curdir) == _HERE:
    sys.path.pop(0)

PROMPTS = ["Write a Python LRU cache class with get/put and unit tests. Code only.",
           "Write a Python function to parse ISO-8601 durations, with tests. Code only.",
           "Write a Python trie with insert, search and prefix listing, with tests. Code only.",
           "Write a Python token-bucket rate limiter class with tests. Code only."]


def check_import_tree(exllamav3):
    """Fail unless exllamav3 came from the first PYTHONPATH entry (the checkout under test)."""
    pp = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app")]
    if not pp:
        sys.exit("FAIL: set PYTHONPATH to the checkout under test (e.g. /path/to/exllamav3-rdna4)")
    tree = os.path.realpath(pp[0])
    loaded = os.path.realpath(exllamav3.__file__)
    if not loaded.startswith(tree + os.sep):
        sys.exit(f"FAIL: exllamav3 imported from {loaded}, not from the PYTHONPATH checkout {tree}")
    return tree


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"A: {a_path} ({a['meta']})\nB: {b_path} ({b['meta']})")
    # Same prompt set on both sides, every prompt present, every output non-empty
    n = len(PROMPTS)
    problems = []
    for tag, r in (("A", a), ("B", b)):
        if r.get("prompts") != PROMPTS:
            problems.append(f"{tag}: prompt set differs from this script's ({len(r.get('prompts') or [])} prompts)")
        if len(r["ids"]) != n:
            problems.append(f"{tag}: {len(r['ids'])} outputs, expected {n}")
        if any(len(x) == 0 for x in r["ids"]):
            problems.append(f"{tag}: empty output(s)")
    if problems:
        print("FAIL: " + "; ".join(problems))
        return False
    ok = True
    for i, (x, y) in enumerate(zip(a["ids"], b["ids"])):
        n = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        same = x == y
        ok &= same
        print(f"  prompt {i}: {'IDENTICAL' if same else 'DIFFER'}  ({len(x)} vs {len(y)} tokens, "
              f"common prefix {n})")
    print("PASS: identical" if ok else "FAIL: outputs differ")
    return ok


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--model", default = os.path.join(root, "models/Qwen3.8-27B-EXL3-3.0bpw"))
    ap.add_argument("--draft", default = os.path.join(root, "models/Qwen3.8-27B-DFlash2-EXL3-3.0bpw"))
    ap.add_argument("--tokens", type = int, default = 400)
    ap.add_argument("--draft-tokens", type = int, default = 7)
    ap.add_argument("--bsz", type = int, default = 1)
    ap.add_argument("--cache-tokens", type = int, default = 16384)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs = 2)
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    assert args.out, "--out or --compare"

    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    import torch
    import exllamav3
    tree = check_import_tree(exllamav3)
    print(f"exllamav3 from {os.path.dirname(exllamav3.__file__)} (PYTHONPATH checkout {tree})")
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
    from exllamav3.cache import CacheLayer_quant

    dcfg = Config.from_directory(args.draft)
    draft = Model.from_config(dcfg)
    kw = dict(layer_type = CacheLayer_quant, k_bits = 4, v_bits = 4, max_batch_size = max(args.bsz, 1),
              max_history = args.draft_tokens)
    dcache = Cache(draft, max_num_tokens = args.cache_tokens, **kw)
    draft.load()
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache_tokens, **kw)
    model.load()
    tok = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, draft_model = draft, draft_cache = dcache, tokenizer = tok,
                    max_batch_size = max(args.bsz, 1), num_draft_tokens = args.draft_tokens,
                    recurrent_cache_size = 512 * 1024**2)

    prompts = PROMPTS
    # The system prompt keeps every prompt above 32 tokens: a 9..32-token prefill now takes the WMMA GEMV
    # (before the batching series: the cooperative GEMM), which rounds differently by design. Between two trees
    # whose decode / verify at bs1 (8 rows) runs the same kernels, the outputs must then match exactly. Across
    # kernel changes (e.g. the dense-row split-decode attention), expect divergences at near-ties only
    system = ("Think efficiently. Work through the problem once, carefully. If you notice yourself re-deriving "
              "the same result, second-guessing a conclusion you already checked, or cycling between options, "
              "stop and commit to the best-supported answer. A confident best answer beats an unfinished analysis.")
    def ids(p):
        t = tok.encode(f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n"
                       f"<|im_start|>assistant\n<think>\n\n</think>\n\n", encode_special_tokens = True)
        assert t.shape[-1] > 32, t.shape
        return t

    def run(batch):
        jobs = [Job(input_ids = ids(p), max_new_tokens = args.tokens, sampler = ArgmaxSampler(), identifier = i)
                for i, p in batch]
        out = {i: [] for i, _ in batch}
        for j in jobs:
            gen.enqueue(j)
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                    out[r["identifier"]] += r["token_ids"].view(-1).tolist()
        return out

    # Warm up graphs and autotune at this shape first, so the measured runs see the steady-state kernels
    run([(0, prompts[0])])
    results, t0 = {}, time.time()
    indexed = list(enumerate(prompts))
    with torch.inference_mode():
        for k in range(0, len(indexed), max(args.bsz, 1)):
            results.update(run(indexed[k:k + max(args.bsz, 1)]))
    wall = time.time() - t0
    n = sum(len(v) for v in results.values())
    meta = (f"{os.path.dirname(exllamav3.__file__)}, bsz {args.bsz}, draft {args.draft_tokens}, "
            + ", ".join(f"{k} {os.environ.get(k, 'unset')}" for k in ("EXL3_GEMV_WMMA", "EXL3_DEC_GROUP", "EXL3_DEC_Q4W", "EXL3_GRAPHS"))
            + f", {n} tokens in {wall:.1f}s")
    json.dump({"meta": meta, "prompts": prompts, "ids": [results[i] for i in range(len(prompts))]},
              open(args.out, "w"))
    print(f"wrote {args.out}: {meta}")
    for i in range(len(prompts)):
        print(f"--- prompt {i}: {len(results[i])} tokens\n{tok.decode(torch.tensor(results[i]))[:300]}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
