"""GPU availability check.

These tests skip cleanly on CPU-only machines (or CPU torch builds), so the
suite stays green everywhere. When a CUDA-enabled torch build *is* installed on
a machine with a visible GPU, they assert that the GPU is actually usable, not
just that the flag is set.
"""
import pytest

try:
    import torch
except Exception as exc:  # ImportError, or OSError from a broken/partial install
    pytest.skip(f"torch not importable: {exc}", allow_module_level=True)

cuda_available = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not cuda_available,
    reason="No CUDA GPU available (CPU-only machine or CPU torch build).",
)


def test_report_torch_build(capsys):
    """Always runs: prints what torch sees, so `pytest -s` documents the setup."""
    with capsys.disabled():
        print(f"\n  torch {torch.__version__}")
        print(f"  built with CUDA: {torch.version.cuda}")
        print(f"  cuda available:  {cuda_available}")
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")


@requires_cuda
def test_gpu_is_visible():
    assert torch.cuda.device_count() >= 1
    assert torch.version.cuda is not None  # CUDA-enabled build, not CPU wheel


@requires_cuda
def test_gpu_can_compute():
    """Move data to the GPU, run an op there, and verify the result is correct."""
    device = torch.device("cuda")
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    c = a @ b
    torch.cuda.synchronize()

    assert c.device.type == "cuda"
    # Cross-check the GPU result against the CPU result.
    expected = a.cpu() @ b.cpu()
    assert torch.allclose(c.cpu(), expected, atol=1e-3)

