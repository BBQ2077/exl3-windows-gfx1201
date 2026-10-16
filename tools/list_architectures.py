"""Record the architectures this runtime loads and the local GPU memory in supported-architectures.json."""
from pathlib import Path
import json
import torch
from exllamav3.version import __version__
from exllamav3.architecture.architectures import get_architectures
root = Path(__file__).resolve().parents[1]
props = torch.cuda.get_device_properties(0)
data = {'exllamav3_version': __version__, 'architectures': sorted(get_architectures()),
        'gpu': props.name, 'gpu_arch': getattr(props, 'gcnArchName', '').split(':')[0],
        'gpu_total_bytes': props.total_memory}
p = root / 'supported-architectures.json'
p.write_text(json.dumps(data, indent=2), encoding='utf-8')
assert json.loads(p.read_text(encoding='utf-8')) == data
print('CAPABILITIES', len(data['architectures']), __version__, data['gpu'], data['gpu_arch'], round(data['gpu_total_bytes'] / 2**30, 2), 'GiB')
