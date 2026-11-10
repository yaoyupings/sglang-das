#!/bin/bash
# Build libwelm_hip_dec.so (loaded by welm_hip_dec.py via ctypes) from welm_hip_dec.hip.
# Needs the DTK toolchain (hipcc, clang 17 with gfx938 support). Run on the target machine:
#   bash welm_hip_dec_build.sh [outdir]      # default outdir: this directory
# Builds gfx938 (BW1101) and gfx936 (BW1000). The fp8 kernels use gfx938-only fp8 mmac /
# convert builtins: on gfx936 they compile to traps and their launches return -2 (bf16 KV only).
# GFX_ARCHS="gfx938" builds a single arch.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here}"
HIPCC="${HIPCC:-/opt/dtk/bin/hipcc}"
archs=()
for a in ${GFX_ARCHS:-gfx938 gfx936}; do archs+=(--offload-arch="$a"); done
"$HIPCC" -shared -fPIC -O3 -std=c++17 "${archs[@]}" -fno-gpu-rdc \
    -o "$out/libwelm_hip_dec.so" "$here/welm_hip_dec.hip"
echo "built $out/libwelm_hip_dec.so"
