work done so far-
* Resolved Python 3.13 dependency conflicts by building a matched Python 3.10 environment from scratch
* Repaired broken pretrained-weight and dataset links by sourcing files from Zenodo and Hugging Face
* Fixed CUDA/torchaudio version mismatch and patched 3 codebase bugs involving imports, Python 3.10, and PyTorch checkpoints
* Validated end-to-end audio-to-coda generation through a live Gradio interface
* Replaced private dataset-dependent embedding workflow with a custom public-API extraction script, processing ~300 real whale recordings
* Generated batches of synthetic pseudocodas using the model’s masked-token sampling pipeline
* Evaluated synthetic audio quality against real whale recordings using Fréchet Audio Distance (FAD)
