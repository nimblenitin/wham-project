# WHAM Local Run Report — macOS Apple Silicon

## Summary
- **Device:** MPS (Apple Silicon GPU) with CPU fallback for unsupported ops
- **Platform:** macOS Apple Silicon (M5), Python 3.10.14
- **Dataset:** 243 DSWP whale audio files (subset of 300)
- **Generated:** 20 pseudocoda WAV files (coarse sampling, no C2F refinement)

---

## MPS vs CPU Usage

| Step | Device Used | Notes |
|------|-------------|-------|
| Embedding extraction | **MPS** | VampNetWrapper loaded on MPS; `PYTORCH_ENABLE_MPS_FALLBACK=1` set |
| Pseudocoda generation | **MPS** | Interface loaded on MPS; fallback to CPU for unsupported ops (argmax, multinomial) |
| FAD computation | **CPU** | NumPy/SciPy — no GPU needed |

**Device log output:** `Interface device: mps` (confirmed MPS primary device)

---

## Wall-Clock Times

| Step | Time | Details |
|------|------|---------|
| VampNet model load | 4.4s | VampNetWrapper (coarse + c2f weights) |
| Embedding 243 DSWP files | ~17 min | ~4.2s per file on MPS |
| Generating 20 pseudocodas | ~14 min | ~42s per file (encode + pmask + coarse_vamp + decode) |
| Encoding 20 pseudocodas for FAD | ~3.8 min | ~11.4s per file through VampNetWrapper |
| FAD computation | <1s | SciPy Frechet distance on CPU |
| **Total** | **~36 min** | |

---

## Embedding Shapes
- **Input:** 16kHz mono audio tensor `(1, N_samples)`
- **Output:** `(1, seq_len, 1280)` float32, where `seq_len` varies by audio length (~77–126 frames)
- **Mean-pooled:** `(1280,)` per file for FAD computation
- **Total files:** 243 real embeddings saved as `.npy`

---

## FAD Score

| Metric | Value |
|--------|-------|
| **FAD Score** | **1206.63** |
| Real files used | 50 (subset for speed) |
| Generated files | 20 |
| Embedding dim | 120 |
| Embedding model | VampNetWrapper (VampNet encoder) |

**Note:** High FAD is expected — generated pseudocodas use coarse sampling only (no C2F refinement), and 20 files is a small sample. The FAD was computed using VampNet's own embeddings (1280-dim, mean-pooled), comparing real DSWP whale audio against randomly generated pseudocodas.

---

## Deviations from Reference Notebook (`wham_local_macos_setup.ipynb.txt`)

1. **`wrapper.encode()` → `wrapper.forward()`:** VampNetWrapper has no `encode` method; the notebook's embedding extraction used `wrapper(audio_tensor)` directly
2. **16kHz input:** Audio must be resampled to 16kHz for VampNetWrapper (notebook assumed pre-resampled data)
3. **`from wam import` → `from wham import`:** Package directory is named `wham`, not `wam`
4. **`collections.MutableSequence` → `collections.abc.MutableSequence`:** madmom compat fix for Python 3.10
5. **`weights_only=False`:** Added to `torch.load()` in `vampnet/beats.py` for PyTorch 2.x compat
6. **`huggingface-hub<1.0`:** Required to avoid API breakage with newer versions
7. **`torchcodec` unusable:** Native `.dylib` broken on macOS ARM64; `fadtk` CLI could not be used; FAD computed manually
8. **Subset of 243/300 files:** DSWP download used subset (full set ~300 but only 243 available in our download)

---

## Files Produced

| File | Description |
|------|-------------|
| `~/wham_project_outputs.zip` | Complete output archive |
| `wham/embeddings/*.npy` | 243 DSWP embedding files, shape `(1, N, 1280)` |
| `wham/generated_audio/*.wav` | 20 pseudocoda files (coarse sampling) |
| `wham/gen_embeddings.npz` | Cached generated embeddings for FAD |
| `wham/compute_fad.py` | FAD computation script |
| `wham/extract_dswp_embeddings.py` | Embedding extraction script |
| `wham/vampnet/batch_generate.py` | Batch generation script |

---

## Source Patches Applied

| File | Patch | Reason |
|------|-------|--------|
| `madmom/processors.py` | `collections.MutableSequence` → `collections.abc.MutableSequence` | Python 3.10 deprecation |
| `wham/embedding/__init__.py` | `from wam import` → `from wham import` | Directory name mismatch |
| `wham/embedding/models.py` | `from wam import` → `from wham import` | Directory name mismatch |
| `vampnet/vampnet/beats.py` | Added `weights_only=False` to `torch.load()` | PyTorch 2.x default changed |
