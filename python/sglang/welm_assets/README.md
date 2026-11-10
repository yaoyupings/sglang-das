# WeLM release assets

Runtime assets the WeLM-v4.5-80B-A3B fp8 8-card deployment needs besides the
model and (optionally) the tuned rocBLAS library. The launcher script is kept
outside the sglang package (deployment tooling, not part of the inference
framework).

| Path | What | Used by |
|---|---|---|
| `vendor_fa/` | WeLM-patched vendor flash_attn (dim-256 prefill, partial paged attention) | launcher must put it first on `PYTHONPATH` |
| `fa8/flash_attn_hg_cuda.so` | fp8 prefix-prefill FA kernel (P-shift build) | `flashattention_backend.py` (default for `WELM_FA8_MODDIR`) |
| `kvfp8/welm_kv_fp8_scales.json` | static per-layer K/V scales for the fp8_e4m3 KV cache | `WELM_KV_FP8_SCALES` |
| `gemm/tuned_r<N>.csv` | offline TunableOp GEMM picks, one per device ordinal | `PYTORCH_TUNABLEOP_FILENAME` |

GPU architectures (2026-11-10):
- `vendor_fa/flash_attn_2_cuda*.so` is a gfx938 + gfx936 build of flash-attention-cutlass
  branch `welm-938d256` @ 45ce210 (welm + the d256 contiguous-varlen pipeline path enabled on
  gfx938, from upstream 6d4dcba). Built with `FLASH_GFX_ARCHS=gfx938,gfx936 FLASH_BUILD_HG=0`;
  on a local disk this takes about 6 minutes, on NFS it is IO-bound and takes hours. Older builds
  are kept next to it: `*.so.bak_938only_20261108` and `*.so.bak_3arch_20261021`.
- `libwelm_arn.so` and `libwelm_hip_dec.so` are gfx938 + gfx936. gfx936 (BW1000) has no fp8
  instructions and is bf16-only: there ARN quantizes with the software e4m3 path, and the fp8
  decode / prefix kernels are traps that the launchers refuse (return -2).
- `fa8/` and `kvfp8/` are fp8-only (gfx938).

The custom HIP kernels live next to their Python wrappers:
`srt/layers/libwelm_hip_dec.so`, `srt/layers/libwelm_arn.so`,
`srt/layers/oe_host_gather.so`, `srt/layers/attention/welm_pa_part.so`,
`srt/layers/attention/welm_fa256.so`.

Binaries (`*.so`) and `vendor_fa/` are not tracked in git: deploy by copying
the whole `sglang` package directory, not by cloning the repository.

## Deploy

1. Copy the `sglang` package directory into the target's
   `python3.10/dist-packages/`.
2. (Optional) copy the tuned rocBLAS library and set `ROCBLAS_TENSILE_LIBPATH`.
3. Copy the launcher `start_welm_8card.sh` (kept outside the package) to the
   target and run `bash start_welm_8card.sh [PORT] [LOG]`. It locates this
   directory through `import sglang`.

The launcher checks that every asset and the model exist and exits with an
error if anything is missing. It also fails if the log does not show the page
router as active. Keep `LOG` on a local disk.
