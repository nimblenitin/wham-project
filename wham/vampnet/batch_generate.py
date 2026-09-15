"""
Batch pseudocoda generation for the WhAM resume project (local macOS version).

Mirrors the exact pipeline used by vampnet/app.py's _vamp() function,
using the same default parameter values as the Gradio UI sliders.

Run with:
  cd ~/wham_project/wham/vampnet
  PYTORCH_ENABLE_MPS_FALLBACK=1 ../../wham_env/bin/python batch_generate.py \
      --args.load conf/interface.yml --Interface.device mps
"""
from pathlib import Path
import os
import numpy as np
import audiotools as at
import argbind

from vampnet.interface import Interface
from vampnet import mask as pmask

Interface = argbind.bind(Interface)
conf = argbind.parse_args()

AUDIO_DIR = Path(os.path.expanduser("~/wham_project/wham/vampnet/data/testing_data/dswp_subset"))
OUT_DIR = Path(os.path.expanduser("~/wham_project/wham/generated_audio"))
N_SAMPLES = 20
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading interface...")
with argbind.scope(conf):
    interface = Interface()
print(f"Interface device: {interface.device}")

audio_files = sorted(list(AUDIO_DIR.glob("*.wav")))
if N_SAMPLES is not None:
    audio_files = audio_files[:N_SAMPLES]

print(f"Generating {len(audio_files)} pseudocodas...")

for i, audio_path in enumerate(audio_files):
    out_path = OUT_DIR / f"pseudocoda_{audio_path.stem}.wav"
    if out_path.exists():
        print(f"[{i+1}/{len(audio_files)}] Skipping {audio_path.name} (already exists)")
        continue

    print(f"[{i+1}/{len(audio_files)}] Processing {audio_path.name}...")
    try:
        sig = at.AudioSignal(str(audio_path))
        sig = interface.preprocess(sig)

        loudness = sig.loudness()
        print(f"  input loudness: {loudness}")

        z = interface.encode(sig)

        mask = pmask.linear_random(z, 1.0)
        mask = pmask.mask_and(mask, pmask.periodic_mask(z, 3, 1, random_roll=True))
        mask = pmask.dropout(mask, 0.0)
        mask = pmask.codebook_unmask(mask, 0)
        mask = pmask.codebook_mask(mask, 9)

        zv = interface.coarse_vamp(
            z,
            mask=mask,
            sampling_steps=36,
            mask_temperature=15.0,
            sampling_temperature=1.0,
            return_mask=False,
            gen_fn=interface.coarse.generate,
            sample_cutoff=0.5,
        )

        sig_out = interface.to_signal(zv).cpu()
        sig_out = sig_out.normalize(loudness)
        sig_out.write(str(out_path))
        print(f"  Saved to {out_path}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

print(f"Done! Generated audio saved to {OUT_DIR}")
