#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default backend is native, can be overridden with LUMINAL_BACKEND env var
export LUMINAL_BACKEND="${LUMINAL_BACKEND:-native}"

# Use Python 3.12 (PyO3 0.23 doesn't support Python 3.14)
PYTHON_VERSION="3.12"

echo "Installing maturin and pytest..."
uv pip install --python $PYTHON_VERSION maturin pytest

echo ""
echo "Building luminal_python with maturin (Python $PYTHON_VERSION)..."
uv run --python $PYTHON_VERSION maturin develop --release

echo ""
echo "Running pytest with backend: $LUMINAL_BACKEND"
uv run --python $PYTHON_VERSION pytest tests -v "$@"
