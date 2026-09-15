"""Compute FAD — reuse cached real embeddings, only encode generated audio."""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/wham_project/wham/wham/embedding"))
sys.path.insert(0, os.path.expanduser("~/wham_project/wham"))

import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from scipy.linalg import sqrtm

from models import VampNetWrapper

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

VAMPNET_MODELS = Path(os.path.expanduser("~/wham_project/wham/vampnet/models"))
GEN_DIR = Path(os.path.expanduser("~/wham_project/wham/generated_audio"))
CACHE = Path(os.path.expanduser("~/wham_project/wham/gen_embeddings.npz"))

print("Loading VampNetWrapper...")
t0 = time.time()
wrapper = VampNetWrapper(
    device=DEVICE,
    coarse=VAMPNET_MODELS / "coarse.pth",
    c2f=VAMPNET_MODELS / "c2f.pth",
    coarse_lora=None, c2f_lora=None, codec_only=False,
)
wrapper.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

def get_embedding(audio_path):
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = wrapper(audio_tensor)
    return emb.cpu().numpy().squeeze(0).mean(axis=0)  # (1280,)

# Load cached real embeddings (already extracted)
real_emb_dir = Path(os.path.expanduser("~/wham_project/wham/embeddings"))
real_embs = []
for f in sorted(real_emb_dir.glob("*.npy"))[:50]:
    real_embs.append(np.load(f).squeeze(0).mean(axis=0))
real_embs = np.array(real_embs)
print(f"Real embeddings loaded: {real_embs.shape}")

# Encode generated audio (skip if cached)
if CACHE.exists():
    print("Loading cached generated embeddings...")
    data = np.load(CACHE)
    gen_embs = data["embs"]
else:
    print("Encoding 20 generated pseudocodas...")
    gen_embs = []
    for f in sorted(GEN_DIR.glob("*.wav")):
        print(f"  {f.name}...", end=" ", flush=True)
        t1 = time.time()
        emb = get_embedding(f)
        print(f"{time.time()-t1:.1f}s")
        gen_embs.append(emb)
    gen_embs = np.array(gen_embs)
    np.savez(CACHE, embs=gen_embs)
    print(f"Saved generated embeddings to {CACHE}")
print(f"Generated embeddings: {gen_embs.shape}")

# FAD (Frechet distance)
mu_r, mu_g = real_embs.mean(0), gen_embs.mean(0)
sig_r = np.cov(real_embs, rowvar=False)
sig_g = np.cov(gen_embs, rowvar=False)

diff = mu_r - mu_g
covmean, _ = sqrtm(sig_r @ sig_g, disp=False)
if np.iscomplexobj(covmean):
    covmean = covmean.real

fad = diff @ diff + np.trace(sig_r + sig_g - 2 * covmean)
print(f"\n{'='*50}")
print(f"FAD Score: {fad:.4f}")
print(f"Real: {real_embs.shape[0]} files, Generated: {gen_embs.shape[0]} files")
print(f"Embedding dim: {real_embs.shape[1]}")
