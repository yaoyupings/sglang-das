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
