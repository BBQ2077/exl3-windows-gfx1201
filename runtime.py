"""Process-local bootstrap for the precompiled native Windows runtime."""
from pathlib import Path
import atexit
import ctypes
from ctypes import wintypes
import json
import os
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parent
SITE = ROOT / 'python/Lib/site-packages'
if os.name != 'nt':
    raise SystemExit('Native Windows is required')
for key in list(os.environ):
    if key.startswith(('CUDA', 'HIP', 'ROCM', 'TRITON', 'PYTHON')) or key in {
        'CC', 'CXX', 'INCLUDE', 'LIB', 'LIBPATH', 'VCToolsInstallDir', 'VSINSTALLDIR',
        'VCINSTALLDIR', 'WindowsSdkDir', 'DISTUTILS_USE_SDK', 'TORCH_EXTENSIONS_DIR'}:
        os.environ.pop(key, None)
windows = Path(os.environ.get('SystemRoot', 'C:/Windows'))
os.environ['PATH'] = os.pathsep.join(map(str, [ROOT / 'python', windows / 'System32', windows]))
for key, rel in {
    'TRITON_CACHE_DIR': 'cache/triton', 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
    'TORCHINDUCTOR_CACHE_DIR': 'cache/torchinductor', 'HF_HOME': 'cache/huggingface',
    'TMP': 'tmp', 'TEMP': 'tmp', 'LOCALAPPDATA': 'state/local', 'EXL3_LOG_DIR': 'logs',
}.items():
    path = ROOT / rel
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['EXL3_GRAPHS'] = '0'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1201'
os.environ['TRITON_BACKENDS_IN_TREE'] = '1'
os.environ['TRITON_LIBHIP_PATH'] = str(SITE / '_rocm_sdk_core/bin/amdhip64_7.dll')
os.environ['ROCM_HOME'] = str(SITE / '_rocm_sdk_core')
os.environ['HIP_PATH'] = os.environ['ROCM_HOME']
# DLL search directories stay local, except Windows and the installed GPU driver.
_dll_handles = []
for p in [ROOT / 'python', ROOT / 'native', SITE / 'torch/lib',
          SITE / '_rocm_sdk_core/bin', SITE / '_rocm_sdk_libraries/bin',
          SITE / '_rocm_sdk_core/lib/rocm_sysdeps/bin']:
    if p.is_dir():
        _dll_handles.append(os.add_dll_directory(str(p)))
for p in [ROOT / 'native', ROOT / 'app', ROOT / 'app/examples']:
    sys.path.insert(0, str(p))
for stream in [sys.stdout, sys.stderr]:
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

subprocesses = []
def audit(event, args):
    if event == 'subprocess.Popen':
        subprocesses.append(str(args[1]))
sys.addaudithook(audit)

def loaded_dlls():
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.GetCurrentProcess()
    modules = (wintypes.HMODULE * 8192)()
    needed = wintypes.DWORD()
    psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    if not psapi.EnumProcessModules(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    paths = []
    for m in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        name = ctypes.create_unicode_buffer(32768)
        if psapi.GetModuleFileNameExW(handle, m, name, len(name)):
            paths.append(name.value)
    return sorted(set(paths))

status = {'exit_code': 1}
def write_audit(phase="exit"):
    paths = loaded_dlls()
    security_root = Path(os.environ.get('ProgramData', 'C:/ProgramData')) / 'Microsoft/Windows Defender/Platform'
    security_modules = [p for p in paths if Path(p).is_relative_to(security_root)]
    external_dlls = [p for p in paths if not Path(p).is_relative_to(ROOT) and not Path(p).is_relative_to(windows) and p not in security_modules]
    python_files = sorted(set(str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                              if getattr(m, '__file__', None) and not str(m.__file__).startswith('<') and Path(m.__file__).is_file()))
    external_python = [p for p in python_files if not Path(p).is_relative_to(ROOT)]
    data = {'root': str(ROOT), 'python': sys.executable, 'prefix': sys.prefix,
            'exit_code': status['exit_code'] if phase == 'exit' else None, 'phase': phase, 'loaded_dlls': paths,
            'external_non_windows_dlls': external_dlls, 'windows_security_modules': security_modules, 'external_python_modules': external_python,
            'subprocess_attempts': subprocesses,
            'exllamav3_extension': getattr(sys.modules.get('exllamav3_ext'), '__file__', None),
            'hip_utils_extension': getattr(sys.modules.get('hip_utils'), '__file__', None)}
    data['isolation_pass'] = not external_dlls and not external_python and not subprocesses
    p = ROOT / 'logs' / ('runtime-audit-' + str(os.getpid()) + '.json')
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'logs/last-audit.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('RUNTIME_AUDIT', json.dumps({'file': str(p), 'isolation_pass': data['isolation_pass'],
                                      'exit_code': data['exit_code']}, ensure_ascii=False), flush=True)
atexit.register(write_audit)

try:
    import torch
    import exllamav3_ext
    import hip_utils
    gpu = torch.cuda.get_device_properties(0)
    if gpu.gcnArchName.split(':')[0] != 'gfx1201':
        raise RuntimeError('This runtime targets gfx1201; detected ' + gpu.gcnArchName)
    print('PRECOMPILED_RUNTIME', gpu.name, torch.__version__, flush=True)
    args = sys.argv[1:]
    command = args.pop(0) if args else 'chat'
    if command in {'chat', 'qwen'}:
        if command == 'qwen':
            config = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
            def model_path(key):
                p = Path(config[key])
                return str(p if p.is_absolute() else ROOT / p)
            args = ['-m', model_path('main'), '-dm', model_path('draft'), '-mode', 'qwen35',
                    '-cs', '32768', '-cq', '4', '-rcs', '4', '-ndt', '7', '-chunk_size', '1024',
                    '-gcs', '512', '-maxr', '0', '-think', '-tps'] + args
        sys.argv = [str(ROOT / 'app/examples/chat.py')] + (args or ['--help'])
        runpy.run_path(sys.argv[0], run_name='__main__')
    elif command == 'verify':
        sys.argv = [str(ROOT / 'tests/runtime_probe.py')] + args
        runpy.run_path(sys.argv[0], run_name='__main__')
    elif command == 'script':
        path = (ROOT / args.pop(0)).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError('Scripts must be inside the runtime folder')
        sys.argv = [str(path)] + args
        runpy.run_path(str(path), run_name='__main__', init_globals={'runtime_write_audit': lambda: write_audit('ready')})
    else:
        raise ValueError('Usage: runtime.py [qwen | chat <model args> | verify | script <file>]')
    status['exit_code'] = 0
except SystemExit as e:
    status['exit_code'] = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    raise
except BaseException:
    traceback.print_exc()
    raise SystemExit(1)
