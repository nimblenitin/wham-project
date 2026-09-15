import sys, os
sys.path.insert(0, os.path.expanduser("~/wham_project/wham/wham/embedding"))
sys.path.insert(0, os.path.expanduser("~/wham_project/wham"))

import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

from models import VampNetWrapper

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("Using device:", DEVICE)

VAMPNET_MODELS = Path(os.path.expanduser("~/wham_project/wham/vampnet/models"))
AUDIO_DIR = Path(os.path.expanduser("~/wham_project/wham/vampnet/data/testing_data/dswp_subset"))
OUT_DIR = Path(os.path.expanduser("~/wham_project/wham/embeddings"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading VampNetWrapper...")
wrapper = VampNetWrapper(
    device=DEVICE,
    coarse=VAMPNET_MODELS / "coarse.pth",
    c2f=VAMPNET_MODELS / "c2f.pth",
    coarse_lora=None,
    c2f_lora=None,
    codec_only=False,
)
wrapper.eval()

audio_files = sorted(list(AUDIO_DIR.glob("*.wav")))
print(f"Found {len(audio_files)} audio files")

for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
    out_path = OUT_DIR / f"{audio_path.stem}.npy"
    if out_path.exists():
        continue
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # VampNetWrapper expects 16kHz
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = wrapper(audio_tensor)
        emb_np = emb.cpu().numpy()
        np.save(out_path, emb_np)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

print(f"Done! Embeddings saved to {OUT_DIR}")
