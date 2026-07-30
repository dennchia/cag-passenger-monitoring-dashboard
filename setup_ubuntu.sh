#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo apt update
sudo apt install -y \
  python3-venv \
  python3-tk \
  libgl1 \
  libglib2.0-0 \
  mosquitto \
  mosquitto-clients

python3 -m venv "$ROOT/.venv-cv-linux"
"$ROOT/.venv-cv-linux/bin/python" -m pip install --upgrade pip
"$ROOT/.venv-cv-linux/bin/python" -m pip install -r "$ROOT/edge_tracker/requirements-ubuntu.txt"

python3 -m venv "$ROOT/backend/.venv-linux"
"$ROOT/backend/.venv-linux/bin/python" -m pip install --upgrade pip
"$ROOT/backend/.venv-linux/bin/python" -m pip install -r "$ROOT/backend/requirements.txt"

NODE_MAJOR=0
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
fi
if ! command -v npm >/dev/null 2>&1 || [[ "$NODE_MAJOR" -lt 18 ]]; then
  NVM_DIRECTORY="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$NVM_DIRECTORY/nvm.sh" ]]; then
    export NVM_DIR="$NVM_DIRECTORY"
    # shellcheck source=/dev/null
    source "$NVM_DIR/nvm.sh"
    nvm use 20 --silent >/dev/null
    NODE_MAJOR="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
  fi
fi
if ! command -v npm >/dev/null 2>&1 || [[ "$NODE_MAJOR" -lt 18 ]]; then
  echo "Node.js 20 is required before frontend setup. Install it with nvm, then rerun this script." >&2
  exit 1
fi

npm --prefix "$ROOT/frontend" install

echo
echo "Ubuntu dependencies installed."
echo "PyTorch is intentionally not installed by this script because its command depends on your NVIDIA driver/CUDA setup."
echo "Install the CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ into .venv-cv-linux."
echo "Then run: bash launch_tracker_ubuntu.sh"
