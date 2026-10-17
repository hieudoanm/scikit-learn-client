#!/usr/bin/env bash
set -euo pipefail

BINARY_NAME="skl"
INSTALL_DIR="${HOME}/.local/bin"
DOWNLOAD_URL="https://github.com/hieudoanm/scikit-learn-client/releases/download/latest/skl"

mkdir -p "$INSTALL_DIR"

echo "Downloading $BINARY_NAME from $DOWNLOAD_URL ..."
curl -fsSL "$DOWNLOAD_URL" -o "$INSTALL_DIR/$BINARY_NAME"

chmod +x "$INSTALL_DIR/$BINARY_NAME"

echo "Installed $BINARY_NAME to $INSTALL_DIR/$BINARY_NAME"
echo "Ensure $INSTALL_DIR is on your PATH."
