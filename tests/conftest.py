import pytest


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch):
    monkeypatch.delenv("GPUCHECK_API_KEY", raising=False)
