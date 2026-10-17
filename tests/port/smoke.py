import torch, exllamav3_ext as ext, exllamav3
print("ext file:", ext.__file__)
print("torch", torch.__version__, "hip", torch.version.hip, "gpu", torch.cuda.get_device_name(0))
names = [n for n in dir(ext) if not n.startswith("_")]
print("ext functions:", len(names))
for n in ["pinned_cuda_view", "exl3_gemm", "exl3_mgemm"]:
    print(" ", n, n in names)
from exllamav3.modules.attention_fn import triton_paged
print("QC_STAGING_SLICED:", triton_paged._qc_staging_sliced, "PAGES:", triton_paged._qc_staging_pages)
x = torch.randn(64, 64, device="cuda", dtype=torch.half)
print("gpu matmul ok:", float((x @ x).float().abs().mean()) > 0)
