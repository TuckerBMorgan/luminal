import os

import pytest


@pytest.fixture
def backend():
    """Backend for torch.compile: 'native' or 'cuda'."""
    return os.environ.get("LUMINAL_BACKEND", "native")


@pytest.fixture
def sample_onnx_path(tmp_path):
    """Fixture that could provide a path to a test ONNX model."""
    # TODO: Create or provide a simple test ONNX model
    return None
