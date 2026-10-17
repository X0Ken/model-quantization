#!/usr/bin/env bash
# Run inside a compatible Ascend quantization environment.
set -euo pipefail
REPO=${1:?Usage: bash setup_msmodelslim.sh /absolute/path/to/msmodelslim}
EXPECTED=c61748f295dc8778a978fbf97ee790609f9ed2ea
if [ ! -e "$REPO" ]; then
  git clone --branch deepseek-v41-w4a8 --single-branch https://github.com/X0Ken/msmodelslim.git "$REPO"
fi
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" || {
  echo "Expected patched tool commit $EXPECTED; refusing to change existing checkout" >&2
  exit 1
}
REPO=$(cd "$REPO" && pwd)
mkdir -p "$REPO/msmodelslim/config"
cp "$REPO/config/config.ini" "$REPO/msmodelslim/config/config.ini"
ln -sfn ../config "$REPO/msmodelslim/config_repo"
ln -sfn ../lab_practice "$REPO/msmodelslim/lab_practice"
ln -sfn ../lab_calib "$REPO/msmodelslim/lab_calib"
PYTHON=${PYTHON:-python3}
"$PYTHON" -m pip install -r "$REPO/requirements.txt"
"$PYTHON" -m pip install -e "$REPO"
