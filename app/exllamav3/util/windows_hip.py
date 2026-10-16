"""Process-local compatibility for PyTorch HIP extension translation on Windows."""
import os


def compiler_runtime_library(rocm_home):
    """Locate the compiler-rt builtins supplied with the selected Windows ROCm SDK."""
    from pathlib import Path
    import subprocess
    compiler = Path(rocm_home) / 'lib/llvm/bin/clang-cl.exe'
    runtime_dir = subprocess.check_output(
        [str(compiler), '--print-runtime-dir'], text=True, timeout=10
    ).strip()
    library = Path(runtime_dir) / 'clang_rt.builtins-x86_64.lib'
    if not library.is_file():
        raise FileNotFoundError(library)
    return str(library)


def configure():
    if os.name != 'nt':
        raise RuntimeError('Native Windows is required')
    import torch
    if not torch.version.hip:
        raise RuntimeError('A ROCm-enabled PyTorch build is required')
    from torch.utils.hipify import hipify_python
    if getattr(hipify_python, '_native_windows_paths', False):
        return
    from torch.utils import cpp_extension
    original_library_writer = cpp_extension._write_ninja_file_to_build_library
    def write_library(path, name, sources, *args, **kwargs):
        return original_library_writer(path, name, sorted(sources), *args, **kwargs)
    cpp_extension._write_ninja_file_to_build_library = write_library
    original_write = cpp_extension._maybe_write
    def write_ninja(filename, content):
        if os.path.basename(filename) == 'build.ninja':
            content = ''.join(line.replace(' -O3 ', ' /O2 ') if line.startswith('cflags = ') else line
                              for line in content.splitlines(keepends=True))
        return original_write(filename, content)
    cpp_extension._maybe_write = write_ninja
    original = hipify_python.hipify
    def hipify(*args, **kwargs):
        from pathlib import Path
        files = list(kwargs.get('extra_files', ()))
        for directory in kwargs.get('header_include_dirs', ()):
            for p in Path(directory).rglob('*'):
                if p.is_file() and p.suffix in ('.h', '.hpp', '.cuh'):
                    files.append(str(p))
        kwargs['extra_files'] = list(dict.fromkeys(os.fspath(p).replace('\\', '/') for p in files))
        return original(*args, **kwargs)
    hipify_python.hipify = hipify
    hipify_python._native_windows_paths = True
