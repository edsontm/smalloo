from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from typing import Any, Dict


OMP_TMP_WARNING = 'OMP: Warning #179: Function Can\'t set size of /tmp file failed:'


@dataclass(frozen=True)
class RuntimeProfile:
    platform_system: str
    accelerator: str
    device: str
    distributed_backend: str | None
    num_workers: int
    pin_memory: bool
    amp_enabled: bool
    amp_dtype: str | None
    torch_compile_enabled: bool
    notes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cpu_workers(cpu_count: int | None) -> int:
    if not cpu_count or cpu_count <= 2:
        return 0
    return min(8, max(2, cpu_count - 1))


def _torch_module_available() -> bool:
    return importlib.util.find_spec('torch') is not None


def _filter_native_stderr(captured: str) -> None:
    lines = [line for line in captured.splitlines() if line.strip()]
    forwarded = [line for line in lines if line.strip() != OMP_TMP_WARNING]
    if forwarded:
        os.write(2, ('\n'.join(forwarded) + '\n').encode('utf-8', errors='replace'))


@contextmanager
def _capture_native_stderr() -> Any:
    stderr_fd = os.dup(2)
    with tempfile.TemporaryFile(mode='w+b') as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
            tmp.seek(0)
            captured = tmp.read().decode('utf-8', errors='replace')
            _filter_native_stderr(captured)


def _probe_torch_capabilities() -> Dict[str, Any]:
    if not _torch_module_available():
        return {
            'available': False,
            'cuda_available': False,
            'mps_built': False,
            'mps_available': False,
            'mps_failure_reason': 'torch_not_installed',
        }

    with _capture_native_stderr():
        import torch  # type: ignore

    mps_backend = getattr(torch.backends, 'mps', None)
    mps_built = bool(mps_backend and mps_backend.is_built())
    mps_available = bool(mps_backend and mps_backend.is_available())
    cuda_available = bool(torch.cuda.is_available())
    mps_failure_reason = None

    if mps_built and not mps_available:
        try:
            torch.tensor([1.0], device='mps')
        except Exception as exc:  # pragma: no cover - depends on host runtime
            mps_failure_reason = f'{type(exc).__name__}: {exc}'

    return {
        'available': True,
        'cuda_available': cuda_available,
        'mps_built': mps_built,
        'mps_available': mps_available,
        'mps_failure_reason': mps_failure_reason,
    }


def _has_nvidia_smi() -> bool:
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [executable, '--query-gpu=name', '--format=csv,noheader'],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _build_runtime_profile(
    system_name: str,
    torch_capabilities: Dict[str, Any],
    has_nvidia_smi: bool,
    cpu_count: int | None,
) -> RuntimeProfile:
    notes: list[str] = []
    workers = _cpu_workers(cpu_count)

    if system_name == 'Darwin':
        if torch_capabilities['mps_available']:
            notes.append('Mac detected with Apple MPS available.')
            notes.append('Disable pin_memory on MPS and keep worker count conservative.')
            return RuntimeProfile(
                platform_system=system_name,
                accelerator='mps',
                device='mps',
                distributed_backend=None,
                num_workers=min(workers, 4),
                pin_memory=False,
                amp_enabled=True,
                amp_dtype='float16',
                torch_compile_enabled=False,
                notes=notes,
            )

        notes.append('Mac detected but MPS is unavailable; using CPU fallback.')
        if torch_capabilities.get('mps_built') and torch_capabilities.get('mps_failure_reason'):
            notes.append(f"MPS diagnostic: {torch_capabilities['mps_failure_reason']}")
        elif not torch_capabilities.get('available'):
            notes.append('PyTorch is not installed in the active interpreter.')
        return RuntimeProfile(
            platform_system=system_name,
            accelerator='cpu',
            device='cpu',
            distributed_backend=None,
            num_workers=workers,
            pin_memory=False,
            amp_enabled=False,
            amp_dtype=None,
            torch_compile_enabled=False,
            notes=notes,
        )

    if torch_capabilities['cuda_available'] or has_nvidia_smi:
        notes.append('Nvidia-capable environment detected; prefer CUDA execution.')
        notes.append('Enable pin_memory and prepare NCCL for future distributed runs.')
        return RuntimeProfile(
            platform_system=system_name,
            accelerator='cuda',
            device='cuda',
            distributed_backend='nccl',
            num_workers=workers,
            pin_memory=True,
            amp_enabled=True,
            amp_dtype='float16',
            torch_compile_enabled=True,
            notes=notes,
        )

    notes.append('No Apple MPS or Nvidia CUDA backend detected; using CPU fallback.')
    return RuntimeProfile(
        platform_system=system_name,
        accelerator='cpu',
        device='cpu',
        distributed_backend=None,
        num_workers=workers,
        pin_memory=False,
        amp_enabled=False,
        amp_dtype=None,
        torch_compile_enabled=False,
        notes=notes,
    )


def detect_runtime_profile() -> RuntimeProfile:
    system_name = platform.system()
    torch_capabilities = _probe_torch_capabilities()
    has_nvidia_smi = False if system_name == 'Darwin' else _has_nvidia_smi()
    return _build_runtime_profile(system_name, torch_capabilities, has_nvidia_smi, os.cpu_count())


def apply_runtime_environment() -> RuntimeProfile:
    profile = detect_runtime_profile()

    os.environ['SMALLOO_DEVICE'] = profile.device
    os.environ['SMALLOO_ACCELERATOR'] = profile.accelerator
    os.environ['SMALLOO_NUM_WORKERS'] = str(profile.num_workers)
    os.environ['SMALLOO_PIN_MEMORY'] = '1' if profile.pin_memory else '0'
    os.environ['SMALLOO_AMP_ENABLED'] = '1' if profile.amp_enabled else '0'
    os.environ['SMALLOO_AMP_DTYPE'] = profile.amp_dtype or ''
    os.environ['SMALLOO_TORCH_COMPILE'] = '1' if profile.torch_compile_enabled else '0'

    if profile.distributed_backend:
        os.environ['SMALLOO_DDP_BACKEND'] = profile.distributed_backend
    else:
        os.environ.pop('SMALLOO_DDP_BACKEND', None)

    if profile.accelerator == 'mps':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    elif profile.accelerator == 'cuda':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')

    return profile