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
| Image | `kau/pytorch-master` (CUDA 13, torch 2.11) |
| Language | `Python` |
| Extra modules (if the field exists) | `transformers==5.16.1 peft==0.20.0 pyarrow pyyaml tabulate` — `run.sh` installs them itself when missing |
| Command and GPU | see below |

| Issue | Command | GPU | What it does |
|---|---|---|---|
| 1 smoke | `bash run.sh smoke hf_xxx` | 1 (18 GB slice) | Checks the token against DepthLM and the data repo, downloads and verifies the data pack, downloads the student, trains 30 steps on the bundled 40-image pool, evaluates 3 pixels. ~15 min. Works without a token too (then data and teacher checks are skipped). |
| 2 full chain | `bash run.sh all hf_xxx` | **7** (whole GPU) | Mixed-pool grids (soft, hard) → teacher labeling of the indoor and driving pools → indoor and driving grids (soft, hard). |

Notes.

- `all` runs each stage as a child process; a failed stage does not stop the next one. Re-submitting the same
  command resumes: finished cells, evaluations and label shards are skipped.
- On a whole GPU the script trains 8 cells and evaluates 8 cells concurrently and labels with 4 teacher
  processes (≈30 GB each). On an 18 GB slice everything runs sequentially. Override with `NPROC`, `NPROC_LABEL`.
- Stages can also be submitted one at a time: `bash run.sh grid <mixed|indoor|outdoor> <soft|hard>` and
  `bash run.sh label <indoor|outdoor>`. A grid uses `pools/<pool>/teacher_labels.parquet` from the repository
  if present, otherwise `/app/output/labels/<pool>/teacher_labels.parquet` produced by the labeling stage.
- Stdout is a summary only (the issue report is capped at 65,000 characters); full logs go to `/app/output`.

### Data and token

- The image data is one tar archive split into four parts (6.0 GB) plus `SHA256SUMS`, stored in the **private**
  Hugging Face dataset repo `jh0624/depthlm-distill-data`. On the first run the script downloads it with the token,
  verifies the checksums, extracts it to `/app/output/data/` and deletes the download. Later jobs reuse the extracted
  copy. Alternatively the administrator can place the same files directly under `/app/data/`, which is used first.
  Layout after extraction: `pool/{sunrgbd,kitti,distill_pool}/…` (13,080 training images) and
  `eval/{ibims1,nyuv2,eth3d}/…` (757 images). Ground-truth depth is not shipped as files; the evaluation pixels and
  their depth values are in `ref/`.
- One Hugging Face **read** token, created on the account that accepted the DepthLM license, is passed once as an
  argument of the issue command (`hf_...`). It is used for the data pack and for the gated teacher
  `facebook/DepthLM`; the student `Qwen/Qwen2.5-VL-3B-Instruct` (Apache-2.0) needs no token. At start-up the script
  prints whether the token can reach both repos, so a wrong token is caught in the smoke job. Revoke the token in
  the Hugging Face settings when the chain has finished. Never commit a token: this repository is public.
  (`/app/data/hf_token.txt` is also read if present.)
- Downloaded weights are cached in `/app/output/hf` so later jobs do not download them again. To avoid Hugging Face
  entirely, put local copies under `/app/data/models/Qwen2.5-VL-3B-Instruct` and `/app/data/models/DepthLM`.

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
| `mixed` | 6,400 / 3,697 | indoor 50 %, driving 50 % (matches DepthLM's per-dataset sampling) | included, 44,800 px |
| `indoor` | 6,400 / 5,867 | SUN RGB-D, NYUv2 | produced on the service |
| `outdoor` | 6,400 / 509 | KITTI (scene count saturates at 509; upper N cells add frames of the same drives, reported as a limitation) | produced on the service |

Grid cells: `N400_k1 N400_k4 N400_k16 N1600_k1 N1600_k4 N1600_k16 N6400_k1 N6400_k4`, nested (image order fixed,
pixel indices 0..k−1 shared). Student: Qwen2.5-VL-3B-Instruct + LoRA r16 α32 on q/k/v/o, AdamW 1e-4 cosine,
batch 1 × accumulation 8, 2 epochs, seed fixed. Evaluation: `small` = 300 / 320 / 302 pixels
(iBims-1 / NYUv2 / ETH3D), `large` = 3,000 / 2,000 / 4,503 pixels. Metric: δ1 (max(p/g, g/p) < 1.25).

Measured on an RTX PRO 4500: 0.46 s per training step, ≈10 GB VRAM per cell; teacher labeling 0.8–1.0 s per pixel,
≈30 GB VRAM. H200 timings are to be measured.

## Local run

```bash
pip install -r requirements.txt          # on top of a CUDA build of torch
bash run.sh smoke                        # no data needed
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
- `ref/` and `smoke/data/` contain sparse pixel depth values and a few images from iBims-1, NYU Depth v2, ETH3D,
  SUN RGB-D and KITTI under those datasets' research-use terms. Full datasets are not redistributed.

See `NOTICE` for details.
