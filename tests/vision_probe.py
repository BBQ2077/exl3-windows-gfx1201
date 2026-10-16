"""Vision tower on CPU or GPU: load only the vision component, embed one image, report time and memory.
Usage: python.exe runtime.py script tests/vision_probe.py <model_dir> <cpu|cuda:0> [image]"""
import os
import sys
import time

import torch
from PIL import Image
from exllamav3 import Config, Model, Tokenizer

model_dir, device = sys.argv[1], sys.argv[2]
img_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "app", "examples", "media", "cat.png")
torch.set_num_threads(os.cpu_count() or 8)
with torch.inference_mode():
    config = Config.from_directory(model_dir)
    tok = Tokenizer.from_config(config)
    t0 = time.perf_counter()
    vm = Model.from_config(config, component="vision")
    vm.load(device=device)
    t1 = time.perf_counter()
    gpu_alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    img = Image.open(img_path)
    for i in range(2):
        t2 = time.perf_counter()
        emb = vm.get_image_embeddings(tokenizer=tok, image=img)
        t3 = time.perf_counter()
        print(f"VISION_PROBE device={device} run={i} load_s={t1 - t0:.1f} embed_s={t3 - t2:.2f} "
              f"tokens={emb.embeddings.shape[0]} dtype={emb.embeddings.dtype} "
              f"finite={bool(torch.isfinite(emb.embeddings.float()).all())} gpu_alloc_gib={gpu_alloc:.2f} "
              f"size={img.size}->{emb.metadata.get('preprocessed_size')}", flush=True)
