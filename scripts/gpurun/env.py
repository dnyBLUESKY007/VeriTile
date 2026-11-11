"""Describe the software stack of this machine and compare it with a request."""
import importlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess

# environment name -> (module, text its version must contain, exact match)
PINS = {
    'triton': ('triton', '3.5.1', True),
    'triton-dist': ('triton_dist', 'git7908e4e', False),
    'tilelang': ('tilelang', 'git35850c0', False),
    'tilescale': ('tilelang', 'git71e7145', False),
}
MODULES = ('torch', 'triton', 'triton_dist', 'tilelang', 'nvshmem')


def module_version(name):
    """Version of an importable module or installed distribution, or None."""
    try:
        version = getattr(importlib.import_module(name), '__version__', None)
        if version:
            return str(version)
    except Exception:  # a broken install must not stop the report
        pass
    for distribution in (name, name.replace('_', '-')):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def os_release():
    try:
        for line in Path('/etc/os-release').read_text().splitlines():
            if line.startswith('PRETTY_NAME='):
                return line.split('=', 1)[1].strip('"')
    except OSError:
        pass
    return platform.system()


def gpus():
    """Model and compute capability of each visible GPU, plus driver and CUDA."""
    info = {'gpus': [], 'cuda': None, 'driver': None}
    try:
        import torch
        info['cuda'] = torch.version.cuda
        for index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(index)
            info['gpus'].append({'name': torch.cuda.get_device_name(index), 'sm': major * 10 + minor})
    except Exception:
        pass
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30)
        info['driver'] = out.stdout.splitlines()[0].strip() if out.returncode == 0 and out.stdout else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def collect():
    """Everything env.json reports. No host name, user name or path."""
    return {'python': platform.python_version(), 'os': os_release(), 'kernel': platform.release(),
            'versions': {name: module_version(name) for name in MODULES}, **gpus()}


def mismatches(requires, env):
    """Reasons why this machine must not run a batch with these requirements."""
    reasons = []
    if requires.get('nodes', 1) > 1:
        reasons.append(f"needs {requires['nodes']} nodes; this is one node")
    wanted = requires.get('gpus', 0)
    sm = requires.get('sm')
    usable = [g for g in env.get('gpus', []) if sm is None or g.get('sm') == sm]
    if len(usable) < wanted or (sm is not None and not usable):
        reasons.append(f"needs {max(wanted, 1)} GPU(s)" + (f' with sm_{sm}' if sm is not None else '')
                       + f"; found {[g.get('sm') for g in env.get('gpus', [])]}")
    name = requires.get('env')
    if name is not None:
        if name not in PINS:
            reasons.append(f'unknown environment {name!r}')
        else:
            module, text, exact = PINS[name]
            version = env.get('versions', {}).get(module)
            if version is None or (version != text if exact else text not in version):
                reasons.append(f"environment {name!r} pins {module} {'==' if exact else 'containing'} "
                               f'{text}; found {version}')
    return reasons
