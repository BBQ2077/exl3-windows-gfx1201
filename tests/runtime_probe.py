"""Cold-cache GPU checks for the packaged runtime."""
from pathlib import Path
import json
import torch
import triton
import triton.language as tl
from exllamav3.ext import exllamav3_ext as ext
import hip_utils

@triton.jit
def vector_add(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + i, tl.load(x + i, i < n, 0) + tl.load(y + i, i < n, 0), i < n)

root = Path(__file__).resolve().parents[1]
checks = []
for n, block in [(257, 128), (4097, 256)]:
    x = torch.arange(n, device='cuda', dtype=torch.float32)
    y = torch.ones_like(x)
    out = torch.empty_like(x)
    vector_add[(triton.cdiv(n, block),)](x, y, out, n, block)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    checks.append('triton_vector_add_' + str(n))
a = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 256
actual = a.to('cuda', dtype=torch.float16) @ a.to('cuda', dtype=torch.float16).T
torch.testing.assert_close(actual.float().cpu(), a @ a.T, rtol=0.002, atol=0.002)
checks.append('torch_fp16_matmul')
torch.manual_seed(123)
x = torch.randn((4, 256), device='cuda', dtype=torch.float16)
w = torch.randn((256,), device='cuda', dtype=torch.float16)
y = torch.empty_like(x)
ext.rms_norm(x, w, y, 1e-6, 0.0, 1.0, False, False, 1)
ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * w.float()).half()
torch.cuda.synchronize()
torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)
checks.append('exllamav3_rms_norm')
result = {'status': 'PASS', 'checks': checks, 'torch': torch.__version__, 'hip': torch.version.hip,
          'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(0),
          'extension': ext.__file__, 'hip_utils': hip_utils.__file__,
          'rms_norm_max_absolute_error': (y.float() - ref.float()).abs().max().item()}
p = root / 'logs/runtime-probe.json'
p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
assert json.loads(p.read_text(encoding='utf-8')) == result
print('RUNTIME_PROBE', json.dumps(result, ensure_ascii=False), flush=True)
