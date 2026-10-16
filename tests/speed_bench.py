"""Decode-speed bench: same model_init/Generator setup as server.py, no HTTP.
Usage: runtime.py script tests/speed_bench.py <model_init args> [--thread] [--tokens N] [--server-gen]"""
import argparse
import json
import threading
import time

import torch
from exllamav3 import Generator, GreedySampler, Job, model_init

PROMPTS = {
    "zh": "請寫一篇約 600 字的短文，介紹台灣夜市文化的由來、常見小吃與逛夜市的禮儀。",
    "code": "Write a Python class implementing an LRU cache with get, put and delete methods, plus unit tests.",
    "zhthink": "請寫一篇約 600 字的短文，介紹台灣夜市文化的由來、常見小吃與逛夜市的禮儀。",
    "en": "Write a 600-word essay about Taiwanese night market culture: its origins, popular street foods, "
          "and etiquette for visiting a night market.",
    "enthink": "Write a 600-word essay about Taiwanese night market culture: its origins, popular street foods, "
               "and etiquette for visiting a night market.",
}
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
THINK_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_sampling_args=True, add_draft_model_args=True,
                        default_cache_size=32768, default_recurrent_cache_size=4.0, default_chunk_size=1024)
    parser.add_argument("--thread", action="store_true")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="code")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    with torch.inference_mode():
        model, config, cache, tok, dmodel, _, dcache = model_init.init(args)
        gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dmodel, draft_cache=dcache,
                        num_draft_tokens=args.num_draft_tokens, max_batch_size=1, max_chunk_size=args.chunk_size,
                        cpu_cache_size=0, recurrent_cache_size=int(args.recurrent_cache_size * 1024 ** 3),
                        dynamic_draft_tokens=args.dynamic_draft, draft_confidence=args.draft_confidence,
                        ngram_match_min=args.ngram_match_min, hybrid_draft_tokens=args.hybrid_draft_tokens)
    free, total = torch.cuda.mem_get_info()
    print("MEM_AFTER_LOAD", json.dumps({"free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2),
          "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
          "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)}), flush=True)
    tpl = THINK_TEMPLATE if args.prompt.endswith("think") else TEMPLATE
    ids = tok.encode(tpl.format(PROMPTS[args.prompt]), add_bos=False, encode_special_tokens=True)
    stops = [x for x in (config.eos_token_id_list or []) if x]

    def run(sampler, label):
        with torch.inference_mode():
            gen.enqueue(Job(input_ids=ids, max_new_tokens=args.tokens, sampler=sampler, stop_conditions=stops))
            last = None
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("eos"):
                        last = r
        n, t = last["new_tokens"], last["time_generate"]
        acc, rej = last.get("accepted_draft_tokens"), last.get("rejected_draft_tokens")
        print("BENCH", json.dumps({"cfg": args.label, "run": label, "prompt": args.prompt, "thread": args.thread,
              "cs": args.cache_size, "tokens": n,
              "tps": round(n / t, 1), "draft": None if acc is None else round(acc / max(1, acc + rej), 3)}), flush=True)

    def body():
        for i in range(args.runs):
            run(GreedySampler(), f"greedy{i}")
        run(model_init.get_arg_sampler(args), "sampled")

    if args.thread:
        t = threading.Thread(target=body)
        t.start()
        t.join()
    else:
        body()


if __name__ == "__main__":
    main()
