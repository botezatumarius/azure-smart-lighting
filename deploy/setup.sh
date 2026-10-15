#!/usr/bin/env bash
# Install and start the smart-lighting service on Ubuntu.
# Idempotent: rerun to redeploy.
set -euo pipefail
cd "$(dirname "$0")/.."

# Required packages (python3-opencv also pulls in OpenCV's system libraries).
sudo apt-get update -q
sudo apt-get install -y -q python3-opencv python3-pip python3-venv

# 1 GiB of RAM needs swap for pip and the model.
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
fi

# --no-cache-dir keeps pip's memory use low.
python3 -m venv .venv
.venv/bin/pip install -q --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q --no-cache-dir -e ".[vm]"

# Upload token (created once).
if ! sudo test -f /etc/smart-lighting.env; then
  echo "SMART_LIGHTING_TOKEN=$(openssl rand -hex 16)" | sudo tee /etc/smart-lighting.env > /dev/null
  sudo chmod 600 /etc/smart-lighting.env
fi

sed "s|@USER@|$USER|g; s|@DIR@|$PWD|g" deploy/smart-lighting.service \
  | sudo tee /etc/systemd/system/smart-lighting.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable smart-lighting
sudo systemctl restart smart-lighting

echo "Service started. Upload token:"
sudo cat /etc/smart-lighting.env
