#!/usr/bin/env bash
# Runs once when the Codespace is created (postCreateCommand).
# Mirrors .github/workflows/bot.yml steps 1:1 so behavior matches GitHub Actions.
set -euo pipefail

echo "==> Updating apt and installing system dependencies..."
sudo apt-get update

# skia-python (used by MangaTranslator for text rendering) needs a full
# graphics/EGL stack even headless - libgl1 alone is not enough.
#
# fontconfig + fonts-liberation are added on top of the bot.yml list:
# Codespaces' base devcontainer image ships an incomplete/stub fontconfig
# config (missing font dirs), which is what throws:
#   Fontconfig warning: ".../05-reset-dirs-sample.conf", line 6: unknown element "reset-dirs"
# and, worse, can leave skia-python with zero usable system fonts, causing
# main.py to silently exit without writing any translated images.
# Installing fontconfig properly + rebuilding its cache + adding a real font
# package fixes both the warning and the silent-failure root cause.
sudo apt-get install -y \
  libgl1 \
  libegl1 \
  libgles2 \
  libglx-mesa0 \
  libgl1-mesa-dri \
  libglu1-mesa \
  libxi6 \
  libxrender1 \
  libxrandr2 \
  libxfixes3 \
  libxcursor1 \
  libxdamage1 \
  libxcomposite1 \
  libxkbcommon0 \
  fontconfig \
  fonts-liberation \
  fonts-dejavu-core \
  unzip zip ffmpeg

echo "==> Rebuilding fontconfig cache..."
sudo fc-cache -f -v > /tmp/fc-cache.log 2>&1 || true
fc-list | head -5 || echo "WARNING: fc-list returned no fonts - font rendering may fail."

echo "==> Cloning MangaTranslator core (pinned v1.21.0, same as bot.yml)..."
rm -rf MangaTranslator
git clone --branch v1.21.0 --depth 1 https://github.com/meangrinch/MangaTranslator.git

cd MangaTranslator

echo "==> Installing PyTorch (CPU build - Codespaces has no GPU)..."
pip install torch==2.11.0 torchvision==0.26.0 --extra-index-url https://download.pytorch.org/whl/cpu

echo "==> Installing MangaTranslator requirements..."
pip install -r requirements.txt

echo "==> Installing bot runtime dependencies..."
pip install Pillow numpy openai pyrogram tgcrypto PyMuPDF

cd ..

echo "==> Verifying core imports..."
python -c "import fitz; import PIL; import numpy; import pyrogram; import tgcrypto; print('core imports OK')"

echo ""
echo "=================================================================="
echo " Setup complete."
echo ""
echo " Before running the bot, set these env vars (Codespaces secrets"
echo " recommended: repo/org Codespaces secrets, or export manually):"
echo "   export API_ID=..."
echo "   export API_HASH=..."
echo "   export BOT_TOKEN=..."
echo "   export AUTHORIZED_USERS=..."
echo "   export HF_TOKEN=...          # for OSB model download from HF Hub"
echo "   export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN"
echo ""
echo " Then run: python bot.py"
echo "=================================================================="
