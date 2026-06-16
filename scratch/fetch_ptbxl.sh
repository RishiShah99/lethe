#!/usr/bin/env bash
# Fetch the open-access PTB-XL 1.0.3 dataset onto the box into a layout the
# medical/data.py PTBXL loader expects (<root>/ptbxl_database.csv, records100/...).
# Tries the single static zip first (fast), falls back to the recursive mirror.
set -euo pipefail
ROOT="${1:-$HOME/data/ptbxl}"
mkdir -p "$ROOT"
cd "$ROOT"

if [ -f ptbxl_database.csv ]; then
  echo ">> PTB-XL already present at $ROOT"
  echo PTBXL_FETCH_OK
  exit 0
fi

command -v unzip >/dev/null 2>&1 || sudo apt-get install -y -q unzip 2>&1 | tail -1

ZIP_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
echo ">> downloading PTB-XL zip (~1.7 GB)"
if wget -q -O ptbxl.zip "$ZIP_URL"; then
  echo ">> unzipping"
  unzip -q -o ptbxl.zip
  rm -f ptbxl.zip
else
  echo ">> zip download failed; falling back to recursive mirror"
  rm -f ptbxl.zip
  wget -q -r -N -c -np "https://physionet.org/files/ptb-xl/1.0.3/" -P .
fi

# Normalise: hoist the dir containing the index csv up to $ROOT.
D=$(find . -name ptbxl_database.csv -printf '%h\n' | head -1)
if [ -n "$D" ] && [ "$D" != "." ]; then
  shopt -s dotglob
  mv "$D"/* .
  shopt -u dotglob
fi

ls ptbxl_database.csv scp_statements.csv >/dev/null
du -sh records100 2>/dev/null || true
echo ">> PTB-XL ready at $ROOT"
echo PTBXL_FETCH_OK
