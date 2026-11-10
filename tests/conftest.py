import pytest


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch):
    monkeypatch.delenv("GPUCHECK_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_cuda_numbering(monkeypatch):
    # The tool refuses CUDA_VISIBLE_DEVICES on a multi-GPU machine and rewrites
    # CUDA_DEVICE_ORDER; start every test without them and restore them afterwards.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
