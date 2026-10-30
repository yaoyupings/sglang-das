#!/bin/bash
# Build libwelm_hip_dec.so (loaded by welm_hip_dec.py via ctypes) from welm_hip_dec.hip.
# Needs the DTK toolchain (hipcc, clang 17 with gfx938 support). Run on the target machine:
#   bash welm_hip_dec_build.sh [outdir]      # default outdir: this directory
# Only gfx938 (BW1101) is built; the kernels use gfx938 mmac / fp8 builtins.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here}"
HIPCC="${HIPCC:-/opt/dtk/bin/hipcc}"
"$HIPCC" -shared -fPIC -O3 -std=c++17 --offload-arch=gfx938 -fno-gpu-rdc \
    -o "$out/libwelm_hip_dec.so" "$here/welm_hip_dec.hip"
echo "built $out/libwelm_hip_dec.so"
