# depthlm-distill-h200

Distilling the metric depth ability of DepthLM (Pixtral-12B) into a Qwen2.5-VL-3B student with LoRA,
packaged to run unattended on the Aerodrone H200 service (GitHub issue → Jenkins → container).

Two questions are studied on three image pools (indoor, driving, mixed 50/50):

1. **Query-budget allocation.** With the number of teacher queries fixed, is it better to label many images
   with few pixels each or few images with many pixels each? Grid: images N ∈ {400, 1600, 6400} ×
   pixels per image k ∈ {1, 4, 16}, N·k ≤ 25,600 (8 nested cells).
2. **Training signal.** Cross-entropy on the teacher's single answer (`hard`) versus KL to the teacher's
   digit distribution (`soft`).

Everything else is held fixed: pixel lists, teacher labels, steps, initialization, shuffle seed,
label format (one decimal), input focal length 750 and midpoint decoding. ETH3D is fully held out.

Korean version of this page: [docs/README_ko.md](docs/README_ko.md).

## Running on Aerodrone H200

Fill the "container creation and code execution request" issue as follows.

| Field | Value |
|---|---|
| Username | `johnhong06` |
| GitHub link | `https://github.com/johnhong06/depthlm-distill-h200.git` |
| Image | `pytorch/pytorch:latest` (the only PyTorch option in the form). If it is the Docker Hub image (torch 2.2.1), `run.sh` replaces torch with 2.11 (cu128) at start-up, ~5 min; if it already has torch ≥ 2.5 nothing is replaced |
| Language | `Python` |
| Extra modules | none needed in the form: `run.sh` installs `requirements.txt` itself when a module is missing |
| Command and GPU | see below |

| Issue | Command | GPU | What it does |
|---|---|---|---|
| 1 smoke | `bash run.sh smoke hf_xxx` | 1 (18 GB slice) | Installs missing packages (and torch if too old), verifies and extracts the image packs from `/app/data`, reports which teacher and student weights will be used, downloads the student, trains 30 steps on the bundled synthetic 40-image pool, evaluates 3 pixels. ~20 min. Works without a token too (then data and teacher checks are skipped). |
| 2 full chain | `bash run.sh all hf_xxx` | **7** (whole GPU) | Mixed-pool grids (soft, hard) → teacher labeling of the indoor and driving pools → indoor and driving grids (soft, hard). |

Notes.

- `all` runs each stage as a child process; a failed stage does not stop the next one. Re-submitting the same
  command resumes: finished cells, evaluations and label shards are skipped. Labeling only produces the pixels that
  are missing from the repository labels, and a grid refuses to start if any of its pixels lack a label.
- On a whole GPU the script trains 8 cells and evaluates 8 cells concurrently and labels with 4 teacher
  processes (≈30 GB each). On an 18 GB slice everything runs sequentially. Override with `NPROC`, `NPROC_LABEL`.
- Stages can also be submitted one at a time: `bash run.sh grid <mixed|indoor|outdoor> <soft|hard>` and
  `bash run.sh label <indoor|outdoor>`. A grid uses `pools/<pool>/teacher_labels.parquet` from the repository
  if present, otherwise `/app/output/labels/<pool>/teacher_labels.parquet` produced by the labeling stage.
- Stdout is a summary only (the issue report is capped at 65,000 characters); full logs go to `/app/output`.

### Data and weights (no secrets)

The service administrator places these under `/app/data/` once (visible to other users of the service, which is
acceptable for research data and for weights that are distributed with their license); nothing secret is needed
anywhere, so the repository and the request issues can stay public.

| Path under `/app/data/` | Content |
|---|---|
| `depthlm_distill_data.tar.part_aa` … `_ad`, `SHA256SUMS` | image pack, 6.0 GB (13,080 training + 757 evaluation images) |
| `depthlm_distill_data_extra.tar`, `SHA256SUMS_extra` | 205 indoor images added by pool v4, 26 MB |
| `models/DepthLM/` (+ `MODEL_LICENSE`) | teacher weights, 24 GB, FAIR Noncommercial Research License |

- The first run extracts the packs once to `/app/output/data/` and later jobs reuse that copy. Ground-truth depth is
  not shipped as files; the evaluation pixels and their depth values are in `ref/`.
- The student `Qwen/Qwen2.5-VL-3B-Instruct` (Apache-2.0, 7.5 GB) is downloaded from Hugging Face without a token
  and cached in `/app/output/hf`. A local copy under `/app/data/models/Qwen2.5-VL-3B-Instruct` is used if present.
- Fallbacks that need a Hugging Face read token (`hf_…` argument or `/app/data/hf_token.txt`): downloading the image
  packs from the private dataset repo `jh0624/depthlm-distill-data` and the gated teacher from `facebook/DepthLM`.
  The script prints at start-up whether the token can reach both; an invalid token is dropped. Never commit a token.

### Outputs (`/app/output`)

| Path | Content |
|---|---|
| `checkpoints/<cond>_<cell>_<pool>_f750/` | LoRA adapter (`adapter_model.safetensors`), `train.log` |
| `eval/eval_<cond>_<cell>_<pool>_f750[_large].parquet` | Per-pixel prediction, ground truth, uncertainty |
| `tables/table_grid_<cond>_<pool>_f750[_large].md` | δ1 with 95% image-cluster bootstrap CI per cell; row, column and equal-budget paired comparisons |
| `figures/fig_grid_<cond>_<pool>_f750[_large].png` | δ1 versus budget |
| `labels/<pool>/teacher_labels.parquet` | Teacher pseudo-labels produced on the service |
| `results_<cond>_<pool>.zip` | Everything above for one grid, plus logs |
| `run_*.log`, `train_*.log`, `eval_*.log` | Logs |

Ask the administrator for the six zip files and the two label files when the chain finishes.

## Experiments

| Pool | Images / scenes | Domain | Labels |
|---|---|---|---|
| `mixed` | 6,400 / 3,684 | indoor 50 %, driving 50 % (matches DepthLM's per-dataset sampling) | 43,640 px included; 1,160 px (146 images replaced in v4) produced on the service |
| `indoor` | 6,400 / 5,674 | SUN RGB-D, NYUv2 | produced on the service |
| `outdoor` | 6,400 / 509 | KITTI (scene count saturates at 509; upper N cells add frames of the same drives, reported as a limitation) | produced on the service |

Pool v4 (2026-09-23): NYUv2 has ~3 images per room, so the 200 NYUv2 evaluation images' 195 rooms are excluded
from the pools at the scene level (146 mixed-pool and 333 indoor-pool images replaced in place by the next unused
indoor candidates of the same deterministic order; see `pools/*/pool_v4_report.md`). No evaluation image or scene of
NYUv2, iBims-1 or ETH3D is in any pool.

Grid cells: `N400_k1 N400_k4 N400_k16 N1600_k1 N1600_k4 N1600_k16 N6400_k1 N6400_k4`, nested (image order fixed,
pixel indices 0..k−1 shared). Student: Qwen2.5-VL-3B-Instruct + LoRA r16 α32 on q/k/v/o, AdamW 1e-4 cosine,
batch 1 × accumulation 8, 2 epochs, seed fixed. Evaluation: `small` = 300 / 320 / 302 pixels
(iBims-1 / NYUv2 / ETH3D), `large` = 3,000 / 2,000 / 4,503 pixels. Metric: δ1 (max(p/g, g/p) < 1.25).

Measured on an RTX PRO 4500: 0.46 s per training step, ≈10 GB VRAM per cell; teacher labeling 0.8–1.0 s per pixel,
≈30 GB VRAM. H200 timings are to be measured.

## Local run

```bash
bash run.sh smoke                        # installs requirements itself; no data needed
DATA_ROOT=/path/to/data bash run.sh grid mixed soft
```

`DATA_ROOT` must contain `pool/` and `eval/` (or the tar parts). Results go to `./results` unless `OUT_ROOT` is set.

## License and attribution

MIT (`LICENSE`) covers only the code written for this repository. It does not cover the teacher pseudo-labels,
any checkpoint trained from them, the evaluation reference files or the vendored DepthLM code:

- `pools/*/teacher_labels.parquet` and trained adapters are outputs of the DepthLM model and are usable for
  noncommercial research only (FAIR Noncommercial Research License, copy in `third_party/DepthLM_Official/MODEL_LICENSE`).
  Publications must acknowledge DepthLM.
- `third_party/DepthLM_Official/utils/` is vendored from the DepthLM code repository under CC BY-NC 4.0.
- `ref/` contains sparse pixel coordinates and depth values from iBims-1, NYU Depth v2 and ETH3D under those datasets'
  research-use terms. `smoke/data/` images are synthetic. Full datasets are not redistributed.

See `NOTICE` for details.
