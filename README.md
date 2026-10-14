# BigCell: Generating Gigapixel Whole-Slide Images

Official code for **BigCell** (NeurIPS 2026).

[[📄 arXiv]](https://arxiv.org/abs/2610.14261) [[🤗 Models]](https://huggingface.co/StonyBrook-CVLab/BigCell) [[💾 Sample data]](https://huggingface.co/datasets/StonyBrook-CVLab/BigCell-sample-data)

BigCell is a flow-matching transformer that generates 2048×2048 histopathology tiles conditioned on a dense grid
of [UNI-2](https://huggingface.co/MahmoodLab/UNI2-h) embeddings. With an inference-time sliding window it produces
coherent images of up to 32,768×32,768 pixels, and an adaptive window schedule makes this more than 3× faster.
A separate **conditioning generation model (CGM)** generates the UNI-2 grid from a pathology report, which
enables text-to-image generation at 8192×8192.

```
real WSI region ──UNI-2──┐
                         ├──> UNI-2 grid (H/256 × W/256 × 1536) ──> BigCell-large ──> H × W image
report ──TITAN──> CGM ───┘                                          (sliding window over BigCell-tile)
```

## Contents

| Component | Code | Released weights |
|---|---|---|
| BigCell-tile (SiT-XL/2, 845M, Flux.2 latent space) | `bigcell/` | `bigcell-256`, `bigcell-1024`, `bigcell-2048` |
| BigCell-large (sliding window + adaptive scheduling) | `bigcell/sampling.py`, `scripts/generate_from_grid.py` | – |
| CGM (DiT-DH-B, 327M, UNI-2 grid space) | `cgm/` | `cgm-8192` |
| Text → 8k image | `scripts/generate_from_text.py` | – |
| CGM data augmentation | `scripts/augment.py` | – |
| Data preparation | `scripts/extract_features.py`, `scripts/build_manifest.py` | – |

## Installation

```bash
git clone https://github.com/cvlab-stonybrook/BigCell && cd BigCell
pip install -e .
```

BigCell relies on three external models that are **not** redistributed here. Accept their licences on Hugging Face
and log in (`huggingface-cli login`):

- [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev): the VAE (only the `vae/` folder is downloaded)
- [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h): conditioning features (needed only to extract grids from real slides)
- [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN): text encoder for the CGM

Every script accepts a local directory in place of a hub id (`--vae`, `--titan`, `--uni-checkpoint`).
TITAN's remote code requires `transformers==4.56.2`; newer releases produce NaN or incorrect text embeddings.

The BigCell and CGM weights are downloaded automatically from [StonyBrook-CVLab/BigCell](https://huggingface.co/StonyBrook-CVLab/BigCell)
the first time `--model bigcell-2048` / `--cgm cgm-8192` is used. Each `bigcell-*` folder there is also a
diffusers pipeline (`BigCellPipeline`); see the model card for diffusers-only usage.

### Sample data

[StonyBrook-CVLab/BigCell-sample-data](https://huggingface.co/datasets/StonyBrook-CVLab/BigCell-sample-data)
(6.8 GB) holds two 32k regions in the data layout described below. The commands in this README run on it:

```bash
huggingface-cli download StonyBrook-CVLab/BigCell-sample-data --repo-type dataset --local-dir bigcell_sample_data
python scripts/generate_from_grid.py --grid bigcell_sample_data/cptac/AML/C3L-00453-41_x0_y0/8192/0_uni.npy --out out/8k.jpg
```

## Sampling

### From a real slide region

```bash
# 8192×8192 from a 32×32 UNI-2 grid, on one GPU
python scripts/generate_from_grid.py --grid /path/to/region/8192/0_uni.npy --out out/8k.jpg

# 32768×32768, windows sharded across 4 GPUs, adaptive scheduling with r = 0.25 (~3× faster)
torchrun --nproc_per_node 4 scripts/generate_from_grid.py \
    --grid /path/to/region/32768/0_uni.npy --top-r 0.25 --out out/32k.jpg --tiff
```

The output side is 256 × the grid side. Defaults follow the paper: 20 Euler steps, CFG 2.0, 2048 px windows (256 latents) with a stride of 1024 px. 

### From text

```bash
python scripts/generate_from_text.py --out out/pdac --num-samples 2 \
    --prompt "Pancreatic ductal adenocarcinoma, moderately differentiated, with perineural invasion."
```

TITAN encodes the prompt. The CGM samples a 32×32×1536 UNI-2 grid (50-point Euler schedule, CFG 2.0), and
BigCell-large renders it at 8192×8192. The grid is saved next to the image (`sample_<i>_uni.npy`).

### Class-conditional augmentation (Sec. 3.5)

```bash
torchrun --nproc_per_node 4 scripts/augment.py --grid /path/to/region/8192/0_uni.npy \
    --prompt "Clear cell renal cell carcinoma, Fuhrman grade 2." --t-start 0.5 --num-samples 4 --out out/aug
```

### Python API

```python
import numpy as np, torch
from bigcell.models import load_bigcell
from bigcell.vae import Flux2VAE
from bigcell.sampling import sliding_window_sample, tiled_vae_decode

model, cfg = load_bigcell("bigcell-2048", "cuda")
vae = Flux2VAE().cuda()
grid = torch.from_numpy(np.load("8192/0_uni.npy")).float()[None].cuda()   # (1, 32, 32, 1536)
latents, _ = sliding_window_sample(model, grid, alpha_shift=cfg["alpha_shift"], top_r=0.25)
image = tiled_vae_decode(vae, latents)                                     # (8192, 8192, 3) uint8
```

## Data

All data is extracted at **20×**. A *region* is a crop of a slide; `<size>` is its side length in pixels at 20×.

```
<root>/
  <source>/<type>/<slide>_x<X>_y<Y>/<size>/     size ∈ {2048, 4096, 8192, 16384, 32768}
      <N>.jpg          2048×2048 image (the region downsampled when size > 2048)
      <N>_full.jpg     the region at full resolution (size > 2048)
      <N>_uni.npy      (size/256, size/256, 1536) fp16 UNI-2 grid, one token per 256×256 patch
      <N>_vae.npy      (64, 256, 256) fp16 Flux.2 VAE posterior mean ‖ std of <N>.jpg
  metadata/
      image_list_train.h5, image_list_test.h5      datasets size_<size>: relative paths of <N>.jpg
      image_list_text_titan.h5                     {train,test,missing}_<size>: regions with / without a report (CGM)
  image_embedding_stats/uni2_<size>/normalization_stats.pt   per-position mean/var of the UNI-2 grid (CGM)
<titan_text_root>/<source>/<type>/<case>_unpooled.npy   (K, T, 768) TITAN tokens of K report summaries (CGM)
<titan_text_root>/null_embed_unpooled.npy               TITAN tokens of the empty string
```

How each component uses it:

- **BigCell-tile** always trains on `size_2048`. Lower-resolution stages use non-overlapping crops of the
  2048 tiles (with the matching UNI-2 sub-grid).
- **The CGM** trains on the `8192` (or `4096`) grids paired with TITAN report embeddings.
- **Sampling** uses grids of any size.

The paper's training set comprises 15,966,332 tiles from 120,703 WSIs of TCGA (diagnostic and fresh-frozen),
CPTAC, GTEx, HISTAI, NLST and PTRC-HGSOC. The 852,435 test tiles come from held-out slides.

### Preparing data

1. Extract 20× regions from your WSIs (we used [TRIDENT](https://github.com/mahmoodlab/TRIDENT) with 32768 px
   patches). Save them as `<input-root>/<source>/<type>/<slide>_x<X>_y<Y>.png`.
2. Cut multi-size crops and extract features (UNI-2 and Flux.2):
   ```bash
   torchrun --nproc_per_node 4 scripts/extract_features.py --input-root regions/ --out-root bigcell_data/
   ```
3. Build a manifest:
   ```bash
   python scripts/build_manifest.py --root bigcell_data/ --split train
   ```
4. CGM only:
   - Encode report summaries with `cgm.text.encode_and_save` (K paraphrases per case, see Appendix G.1).
   - Compute `normalization_stats.pt`: the per-position mean and variance of the training grids.

## Training

### BigCell-tile (progressive, paper Table 6)

```bash
torchrun --nproc_per_node 8 -m bigcell.train configs/bigcell/256.yaml  --data-dir /path/to/bigcell_data
torchrun --nproc_per_node 8 -m bigcell.train configs/bigcell/512.yaml  --data-dir ... --init-from exps/bigcell_256/checkpoints/<step>.pt
torchrun --nproc_per_node 8 -m bigcell.train configs/bigcell/1024.yaml --data-dir ... --init-from exps/bigcell_512/checkpoints/<step>.pt
torchrun --nproc_per_node 8 -m bigcell.train configs/bigcell/2048.yaml --data-dir ... --init-from exps/bigcell_1024/checkpoints/<step>.pt
```

- Each stage trains for one epoch with AdamW, a 5k-step linear warm-up, EMA 0.9999 and 10% CFG dropout.
- From 1024 on, training uses the SD3 time shift (α = 3), and for the first 5k steps only the conditioning
  embedder and cross-attention are trained (`freeze_steps`).
- Any config value can be overridden on the command line, e.g. `--batch-size`. Training resumes automatically
  from the last checkpoint in `exps/<exp_name>/checkpoints`.
- `--init-from` also accepts a released model directory, e.g. to fine-tune `bigcell-2048`.

### CGM

```bash
torchrun --nproc_per_node 4 -m cgm.train configs/cgm/4096.yaml --data-dir /path/to/bigcell_data --titan-text-dir /path/to/titan_text
torchrun --nproc_per_node 4 -m cgm.train configs/cgm/8192.yaml --data-dir ... --titan-text-dir ... \
    --init-from exps/cgm_4096/checkpoints/ep-XXXXXXX.pt
```

### Converting checkpoints for release

```bash
python scripts/convert_sit_checkpoint.py --ckpt exps/bigcell_2048/checkpoints/0046800.pt --resolution 2048 --alpha-shift 3 --out weights/bigcell-2048
# weights/bigcell-2048 can now be passed to --model / --init-from
python scripts/convert_cgm_checkpoint.py --ckpt exps/cgm_8192/checkpoints/ep-0000080.pt \
    --stats /path/to/bigcell_data/image_embedding_stats/uni2_8192/normalization_stats.pt --out weights/cgm-8192
```

## Citation

```bibtex
@misc{yellapragada2026bigcell,
      title={BigCell: Generating Gigapixel Whole-Slide Images}, 
      author={Srikar Yellapragada and Alexandros Graikos and Zilinghan Li and Kostas Triaridis and Tarak Nath Nandi and Ji Dong K Bai and Beatrice Knudsen and Tahsin Kurc and Prateek Prasanna and Rajarsi R. Gupta and Ravi K Madduri and Joel Saltz and Dimitris Samaras},
      year={2026},
      eprint={2610.14261},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.14261}, 
}
```

## Licence

The code is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `LICENSE`;
third-party code listed in `NOTICE` keeps its original licence). The model weights and sample data are released
under CC BY-NC-ND 4.0. The FLUX.2 VAE, UNI-2 and TITAN are subject to their own licences. BigCell is for
non-commercial research only, and generated images must not be used for clinical diagnosis.
