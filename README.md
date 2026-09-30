# depthlm-distill-h200

Distilling the metric depth ability of DepthLM (Pixtral-12B) into a Qwen2.5-VL-3B student with LoRA, using only the
teacher's answers as labels, packaged to run unattended on the Aerodrone H200 service (GitHub issue → Jenkins →
container). Korean version: [docs/README_ko.md](docs/README_ko.md).

## Study at a glance

| | |
|---|---|
| Question 1: allocation | With the number of teacher queries fixed, is it better to label many images with few pixels each or few images with many pixels each? |
| Question 2: training signal | Cross-entropy on the teacher's single answer (`hard`) or KL to the teacher's digit distribution (`soft`)? |
| Grid | images N ∈ {400, 1600, 6400} × pixels per image k ∈ {1, 4, 16}, N·k ≤ 25,600: 8 nested cells per pool and loss |
| Pools | indoor (SUN RGB-D, NYUv2), driving (KITTI), mixed (50/50), 6,400 images each → 3 pools × 2 losses = 6 grids |
| Labels | DepthLM 12B answers on the pool pixels; ground truth is never used for training |
| Evaluation | each pool on its own domain only: indoor → iBims-1, NYUv2; driving → DDAD, nuScenes; mixed → all four |
| Held fixed | pixel lists, teacher labels, steps, initialization, shuffle seed, label format (one decimal), input focal length 750, midpoint decoding |

Cells that share a budget N·k are compared on the same evaluation pixels, and so are `soft` and `hard` in the same cell.
The decision rule was fixed before any grid finished and is stated under [Results](#results).

## Status

| Pool | Loss | Machine | Pool version | Teacher labels | Training | Evaluation |
|---|---|---|---|---|---|---|
| mixed | soft | local | v5 | committed | cell by cell | 1/8 |
| mixed | hard | local | v5 | committed | cell by cell | 1/8 |
| indoor | soft | H200 | v4 = v5 | committed | done | done (8/8) |
| indoor | hard | H200 | v4 = v5 | committed | next | on H200 |
| outdoor | soft | H200 | v5 | committed (5,640 px added locally) | after indoor hard | local, from the adapters |
| outdoor | hard | H200 | v5 | committed (5,640 px added locally) | | local, from the adapters |

Order of work. On H200, one job at a time on a whole GPU: `grid indoor soft` (running) → `grid indoor hard` →
`grid outdoor soft` → `grid outdoor hard`; after each job, ask the administrator for its `results_<cond>_<pool>.zip`,
which must contain `checkpoints/`. Locally: the new outdoor and mixed teacher labels, then the DDAD and nuScenes teacher
baselines, then `grid mixed soft` and `grid mixed hard` on pool v5, then DDAD and nuScenes for the outdoor adapters from
the H200 zips and the zero-shot student on all four sets.

Pool v5 moves near-duplicate images to the end of the image order ([Pools](#pools)). The indoor pool has none, so its v4
and v5 are the same. A first mixed soft grid trained on v4, before the rule existed, is kept locally as a pilot and not
reported. The H200 archive has no driving sets, so the outdoor grids only train there.

## Results

Every cell is `δ1 / AbsRel` on the large evaluation set, filled in as each cell's evaluation finishes; a dash means the
cell has not been evaluated yet. The numbers come from each cell's evaluation log and are computed exactly as in
`tables/table_grid_<cond>_<pool>_f750_large.md`, which adds the confidence intervals and the paired comparisons.
Cross-domain scores are not computed, since they would measure the domain gap rather than the allocation or the loss.

**Decision rule (fixed on 2026-09-29, before any grid finished).** Every comparison, whether equal-budget cells or
soft against hard in the same cell, is a paired difference in δ1 on the same pixels with a 95 % cluster bootstrap
interval, computed separately on each of the pool's sets. The clusters are scenes on DDAD and nuScenes, which have
five to six images per scene, and images on iBims-1 and NYUv2, where an image is essentially its own scene; on a
synthetic check, scene clusters widen the driving intervals by a factor of 1.5, which image clusters would hide. A result is claimed when all of the pool's sets point the same way and
every interval excludes zero. The same direction with only some intervals excluding zero is reported as weak evidence,
and opposite directions as set-dependent, with no claim. For the mixed pool the rule is applied to the two indoor sets
and to the two driving sets, and a claim for the pool needs both domains to agree. Negative results are reported as well.
`experiments/32_decide.py --pool <pool>` applies the rule to a pool's soft and hard results and writes
`tables/decision_<pool>.md`, with the DDAD sensitivity value (lower quarter of the non-front cameras removed) beside it.

The equal-budget comparisons are N400_k4 vs N1600_k1 at 1,600 queries; N400_k16, N1600_k4 and N6400_k1 at 6,400; and
N1600_k16 vs N6400_k4 at 25,600. The last two rows of each table are baselines rather than cells of the design; they are
single models evaluated once, so they carry no loss condition.

### Mixed pool (indoor 50 / driving 50)

| N | k | Budget | Loss | iBims-1 | NYUv2 | DDAD | nuScenes |
|---:|---:|---:|:--|:--|:--|:--|:--|
| 400 | 1 | 400 | soft | 0.344 / 0.363 | 0.366 / 0.313 | 0.163 / 0.619 | 0.175 / 0.765 |
| | | | hard | 0.338 / 0.360 | 0.393 / 0.307 | 0.159 / 0.646 | 0.183 / 0.711 |
| 400 | 4 | 1600 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| 400 | 16 | 6400 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| 1600 | 1 | 1600 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| 1600 | 4 | 6400 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| 1600 | 16 | 25600 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| 6400 | 1 | 6400 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| 6400 | 4 | 25600 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| **teacher, DepthLM 12B** | | | | 0.811 / 0.141 | 0.889 / 0.122 | 0.652 / 0.240 | 0.581 / 0.694 |
| **student, no distillation** | | | | — | — | — | — |

### Indoor pool

| N | k | Budget | Loss | iBims-1 | NYUv2 |
|---:|---:|---:|:--|:--|:--|
| 400 | 1 | 400 | soft | 0.322 / 0.364 | 0.403 / 0.307 |
| | | | hard | — | — |
| 400 | 4 | 1600 | soft | 0.467 / 0.293 | 0.564 / 0.246 |
| | | | hard | — | — |
| 400 | 16 | 6400 | soft | 0.600 / 0.217 | 0.703 / 0.189 |
| | | | hard | — | — |
| 1600 | 1 | 1600 | soft | 0.461 / 0.297 | 0.539 / 0.252 |
| | | | hard | — | — |
| 1600 | 4 | 6400 | soft | 0.596 / 0.218 | 0.730 / 0.180 |
| | | | hard | — | — |
| 1600 | 16 | 25600 | soft | 0.666 / 0.199 | 0.777 / 0.169 |
| | | | hard | — | — |
| 6400 | 1 | 6400 | soft | 0.602 / 0.216 | 0.724 / 0.181 |
| | | | hard | — | — |
| 6400 | 4 | 25600 | soft | 0.664 / 0.199 | 0.773 / 0.160 |
| | | | hard | — | — |
| **teacher, DepthLM 12B** | | | | 0.811 / 0.141 | 0.889 / 0.122 |
| **student, no distillation** | | | | — | — |

### Driving pool

| N | k | Budget | Loss | DDAD | nuScenes |
|---:|---:|---:|:--|:--|:--|
| 400 | 1 | 400 | soft | — | — |
| | | | hard | — | — |
| 400 | 4 | 1600 | soft | — | — |
| | | | hard | — | — |
| 400 | 16 | 6400 | soft | — | — |
| | | | hard | — | — |
| 1600 | 1 | 1600 | soft | — | — |
| | | | hard | — | — |
| 1600 | 4 | 6400 | soft | — | — |
| | | | hard | — | — |
| 1600 | 16 | 25600 | soft | — | — |
| | | | hard | — | — |
| 6400 | 1 | 6400 | soft | — | — |
| | | | hard | — | — |
| 6400 | 4 | 25600 | soft | — | — |
| | | | hard | — | — |
| **teacher, DepthLM 12B** | | | | 0.652 / 0.240 | 0.581 / 0.694 |
| **student, no distillation** | | | | — | — |

### Baselines

The teacher is `facebook/DepthLM` (12B) scored on exactly the pixels the students are scored on, with the same δ1 and
AbsRel definitions and the same image-cluster bootstrap as `31_grid.py`. Its midpoint correction is +0.005 because the
teacher answers with two decimals, while a student answers with one and gets +0.05. The rows marked *paper* are δ1 values
reported in the DepthLM paper (Table 1) and are quoted, not measured here: they use other random pixels (100 per image,
no seed), all 1,449 NYUv2 images rather than our 200, and z-depth on DDAD and nuScenes, so they place our numbers in
context rather than serving as paired comparisons.

| Baseline | iBims-1 | NYUv2 | DDAD | nuScenes |
|---|:--|:--|:--|:--|
| Teacher, DepthLM 12B, measured | 0.811 / 0.141 | 0.889 / 0.122 | 0.652 / 0.240 (z: 0.677) | 0.581 / 0.694 (z: 0.640) |
| Student before distillation, Qwen2.5-VL-3B, measured | — | — | — | — |
| *paper:* DepthLM 12B (Pixtral) | 0.870 | 0.799 | 0.670 | 0.819 |
| *paper:* DepthLM 3B (Qwen2.5-VL-3B trained on 16M ground-truth images) | 0.890 | 0.868 | 0.724 | 0.870 |
| *paper:* Qwen2.5-VL-3B without training | 0.080 | 0.128 | 0.083 | 0.090 |

95 % image-cluster bootstrap intervals for the measured teacher: iBims-1 [0.763, 0.856], NYUv2 [0.866, 0.911]. On iBims-1
the measured teacher is 0.06 below the paper although inputs, preprocessing and decoding match the official code; on
ETH3D, which is no longer evaluated here, the same pipeline reproduced the paper within 0.009 (0.644 against 0.653).

The zero-shot student row needs one evaluation run with no adapter, which `21_eval_student.py` does when `--adapter` is
left empty. It is the baseline that says how much the distillation added.

## Running on Aerodrone H200

A Korean step-by-step guide to the service itself (application, request form, what the container keeps, data
delivery, pitfalls met so far) is in [docs/h200_guide_ko.md](docs/h200_guide_ko.md), published at
https://johnhong06.github.io/depthlm-distill-h200/. Fill the "container creation and
code execution request" issue as follows.

| Field | Value |
|---|---|
| Username | `johnhong06` |
| GitHub link | `https://github.com/johnhong06/depthlm-distill-h200.git` |
| Image | `pytorch/pytorch:latest` (the only PyTorch option in the form). If it is the Docker Hub image (torch 2.2.1), `run.sh` replaces torch with 2.11 (cu128) at start-up, ~5 min; if it already has torch ≥ 2.5 nothing is replaced |
| Language | `Python` |
| Extra modules | none needed in the form: `run.sh` installs `requirements.txt` itself when a module is missing |
| Command and GPU | see below |

| Command | GPU | What it does |
|---|---|---|
| `bash run.sh check` | smallest slice | Reports, for `/app/data`, `/app/output` and the work root, whether it is writable and how much space is free, and where the archive would be extracted. No GPU, under a minute. |
| `bash run.sh smoke` | 1 (18 GB slice) | Trains 30 steps on the bundled synthetic 40-image pool and evaluates 3 pixels. ~20 min. |
| `bash run.sh label <pool>` | **7** (whole GPU) | Teacher inference for the pixels the committed labels do not cover yet, 4 teacher processes. Writes `labels_<pool>.zip`, whose parquet has to be committed to `pools/<pool>/teacher_labels.parquet`. |
| `bash run.sh grid <pool> <soft\|hard>` | **7** (whole GPU) | Trains the 8 cells concurrently, evaluates them on the pool's own sets that are in the archive (iBims-1 and NYUv2; the driving sets are not, so an outdoor grid only trains), then writes tables, figures and `results_<cond>_<pool>.zip` with the adapters. Training ~12 h, to be measured. |

What the service does and does not keep decides how the jobs are split:

- **Nothing produced by one job is visible to the next** (measured on 2026-09-28: `/app/output`, `/app/scratch` and the
  work volume all start empty). The repository is the only durable channel, so teacher labels are committed to
  `pools/<pool>/teacher_labels.parquet`, and training and evaluation run inside one `grid` job. Separate `train` and
  `eval` jobs only work where the result volume persists, such as a local machine.
- `grid` looks for labels in `/app/output/labels/<pool>/`, then `/app/data/labels/<pool>/` (when `/app/data` is
  writable), then `pools/<pool>/` in the repository, and refuses to train when they do not cover every pixel of the
  grid, naming the pool to label.
- Re-submitting a command resumes: a trained cell, a finished evaluation or a completed label shard is skipped, as long
  as `/app/output` is kept.
- Evaluation covers only the pool's own sets (`EVAL_DATASETS`: indoor → `ibims1 nyuv2`, outdoor → `ddad nuscenes`, mixed
  → all four) and only the large set (`EVAL_SETS=large`; the small set is a subset of it and is read out of it). A set
  that is not under `$DATA_ROOT/eval` is skipped with a message, and the zip still carries the adapters.
- Stdout is a summary only (the issue report is capped at 65,000 characters); full logs go to `/app/output`.

### Outputs (`/app/output`)

| Path | Content |
|---|---|
| `checkpoints/<cond>_<cell>_<pool>_f750/` | LoRA adapter (`adapter_model.safetensors`), `train.log` |
| `eval/eval_<cond>_<cell>_<pool>_f750_large__<set>.parquet` | Per-pixel prediction, ground truth, uncertainty (older runs: one file per cell without `__<set>`) |
| `tables/table_grid_<cond>_<pool>_f750_large.md` | δ1 with 95 % cluster bootstrap interval per cell (scenes on DDAD and nuScenes, images otherwise); row, column and equal-budget paired comparisons |
| `figures/fig_grid_<cond>_<pool>_f750_large.png` | δ1 versus budget |
| `results_<cond>_<pool>.zip` | Everything above for one grid, plus logs |
| `run_*.log`, `train_*.log`, `eval_*.log` | Logs |

After each grid job, ask the administrator for that job's `results_<cond>_<pool>.zip`, and check that it contains
`checkpoints/`: the outdoor adapters are evaluated locally, and any set added later is evaluated from these files.

<details>
<summary>Data archive, storage and sizes on the service</summary>

### Data and weights (no secrets)

Everything the jobs need is one 30 GB tar, split into sixteen 2 GB parts (`depthlm_distill_h200_app_data.tar.part_00`
… `part_15`, plus `SHA256SUMS_parts` and a note), shared with the service administrator through a Google Drive
folder. The administrator only downloads the files into any folder under `/app/data/`, either as the individual parts or
as the zip file(s) that Google Drive produces for a folder download; nothing has to be extracted by hand. `bash run.sh data`
finds the parts (inside the zips if needed), verifies their checksums and extracts them once, next to the parts under
`/app/data`, and every later job reuses that copy. An already extracted `depthlm_distill_h200/` folder or raw
`models/DepthLM/` weights under `/app/data` are used directly if present.

**Nothing is ever extracted into `/app/output`.** That path is for results only. The extraction target is chosen in
this order: next to the parts under `/app/data` when that is writable, so no copy exists at all; then `/app/scratch`,
the pod's work volume; then the container's temporary disk. A target under `/app/output` is refused even when it is
passed explicitly.

Measured on the service on 2026-09-28:

| Path | Writable | Free | Used |
|---|---|---|---|
| `/app/data` (parts in `/app/data/HJ`) | no | 103 GB | 265 GB |
| `/app/output` | yes | 103 GB | 0 GB |
| `/app/scratch` | yes | 950 GB | 0 GB |

`/app/data` and `/app/output` report the same free space, so they share one volume with 103 GB left. That is the volume
the 2026-09 run filled. `/app/scratch` is a separate and far larger volume, and it is where the pod clones the
repository and points `HF_HOME`, so that is where the archive goes. No write access on `/app/data` is needed.

Extraction is unavoidable, because both `transformers` and the image loader read files by path, but the full 30 GB is
not always needed. On a volume that survives between jobs the archive is unpacked once in full and every stage reuses
it. On the container's temporary disk, where each job would otherwise unpack 30 GB again, only the folders the stage
actually reads are extracted:

| Stage | Reads | Extracted on a temporary disk |
|---|---|---|
| `label` | pool images, teacher weights | 29.5 GB |
| `train` | pool images | 5.5 GB |
| `eval` | evaluation set | 0.4 GB |

`/app/scratch` does not survive between jobs. Measured on 2026-09-28: one job extracted the archive to
`/app/scratch/h200_work`, and the next job found nothing there and extracted it again. The repository is also cloned
into a per-job folder and `peft` has to be installed again every time, so treat every job as starting from an empty
work volume.

`/app/output` does not carry over either. Measured on 2026-09-28: one job wrote the indoor teacher labels there, and
an hour later the next job saw `/app/output` empty, holding only its own report file. Each job gets a fresh result
volume, and the administrator retrieves that job's files afterwards.

**Nothing produced by one job is visible to the next.** The repository is the only durable channel, so the teacher
labels have to be committed to `pools/<pool>/teacher_labels.parquet` after every labeling job, and training and
evaluation have to run inside one job. Use `grid <pool> <cond>`, not `train` followed by `eval`. The split into
separate `train` and `eval` jobs only works where the result volume survives between jobs, for example on a local
machine.

Unpacking is therefore part of every job, and it is cheap: 30 GB takes about two minutes on this service, so a
training job spends about twenty seconds on its 5.5 GB and an evaluation job about two seconds on its 0.4 GB. There is
no point running `data` more than once; it is only useful as a one-off check that the sixteen parts verify and unpack.

| Path inside the archive | Content |
|---|---|
| `pool/` | 13,285 training images (SUN RGB-D, KITTI, NYUv2), pool v4 |
| `eval/` | 757 evaluation images and lists (iBims-1, NYUv2, ETH3D) |
| `models/DepthLM/` + `models/DepthLM_MODEL_LICENSE.txt` | teacher weights (24 GB) with a copy of the FAIR Noncommercial Research License, as its section 1.b.ii requires when the weights are handed to a third party |
| `extra_done.txt`, `SHA256SUMS_all`, `README_ADMIN.txt` | marker, checksums of every file, note for the administrator |

The archive contains no secret, so the repository and the request issues stay public and nothing has to be revoked
afterwards. Ground-truth depth is not shipped as files; the evaluation pixels and their depth values are in `ref/`.

- The student `Qwen/Qwen2.5-VL-3B-Instruct` (Apache-2.0, 7.5 GB) is downloaded from Hugging Face without a token
  and cached in `$WORK_ROOT/hf`, which is `/app/data/hf` when that is writable and the container's temporary disk
  otherwise. A local copy under `models/Qwen2.5-VL-3B-Instruct` is used if present.
- Fallbacks that need a Hugging Face read token (`hf_…` argument or `/app/data/hf_token.txt`): downloading the image
  packs from the private dataset repo `jh0624/depthlm-distill-data` and the gated teacher from `facebook/DepthLM`.
  Never commit a token.

Sizes, so the quota is never the thing that stops a run:

| What | Where | Size |
|---|---|---|
| Extracted archive (pool images, evaluation set, teacher weights) | `/app/data`, next to the parts | 30 GB |
| Archive parts, deletable after extraction | `/app/data` | 29 GB |
| Student weight cache | `$WORK_ROOT/hf` | 7.5 GB |
| One grid: 8 LoRA adapters, evaluations, tables, figures, zip | `/app/output` | ≈0.9 GB |
| All six grids plus labels | `/app/output` | ≈6 GB |

`bash run.sh check` prints the live numbers and flags anything in `/app/output` that is not a result. `NEED_GB`
(default 32) is the free space the script insists on before it extracts; it refuses early rather than filling the
volume.

</details>

## Experiments

### Pools

| Pool | Images / scenes | Domain | Teacher labels (44,800 px per pool) |
|---|---|---|---|
| `mixed` | 6,400 / 3,684 | indoor 50 %, driving 50 % (matches DepthLM's per-dataset sampling) | 42,256 px from v4 (local, 1,160 of them on the service) + 2,544 px for v5 (local) |
| `indoor` | 6,400 / 5,674 | SUN RGB-D, NYUv2 | produced on the service |
| `outdoor` | 6,400 / 509 | KITTI raw, depth selection, KITTI 2012 and 2015 | 39,160 px from v4 (service) + 5,640 px for v5 (local) |

The pixels per image are fixed in advance (16 per image, uniform in the image away from a 5 % border, seeded by the
image path) and a cell uses the first k of them on the first N images, so every cell of a pool reads the same labels.

Pool v5 (2026-09-29) moves near-duplicate images to the end of the image order (`experiments/07_dedup_pool_order.py`,
reports in `pools/*/dedup_report.md`). Walking the order within each domain, an image goes to the back of its domain
when an image already kept is within dHash distance 10 and above 0.9 grey-level correlation at 32×32; every position is
then refilled from the domain that held it, so the indoor/driving split stays 50/50 at every N. DepthLM subsampled
"highly similar video frames" because they did not help, without stating a rule; this is an explicit version of that
step. It moves 1,365 of the 6,400 outdoor images (raw frames taken while the car was stopped or slow, and KITTI 2012
and 2015 images, which the pool had taken as `_10`/`_11` pairs 0.1 s apart) and 749 driving images of the mixed pool;
the indoor pool has none. Distinct images among the first N = 400 / 1,600 / 6,400 go from 372 / 1,130 / 5,035 to
400 / 1,600 / 5,035 in the outdoor pool and from 391 / 1,388 / 5,651 to 400 / 1,600 / 5,651 in the mixed pool, so the
many-images cells are not handicapped by near-copies. The image set, the pixel coordinates and the N = 6,400 cells
are unchanged; only images that moved into the first 1,600 needed new labels.

The driving pool's 509 scenes overstate the number of distinct places. The scene count treats each KITTI 2012 and KITTI
2015 image as its own scene, but these images were cut from raw drives, and 249 of the 788 in the outdoor pool (165 of
KITTI 2015, 84 of KITTI 2012) show the same place as a raw or depth-selection frame that is also in the pool (dHash
distance ≤ 10, correlation > 0.9); in the mixed pool it is 176 of 788. The campus drives add more repetition, since
they film one place under several drive numbers. This does not leak into any evaluation set, but it means the image
axis N adds fewer new places in the driving pool than the scene count suggests, and it is reported as a limitation
when N and k are compared there.

Pool v4 (2026-09-23): NYUv2 has ~3 images per room, so the 200 NYUv2 evaluation images' 195 rooms are excluded
from the pools at the scene level (146 mixed-pool and 333 indoor-pool images replaced in place by the next unused
indoor candidates of the same deterministic order; see `pools/*/pool_v4_report.md`). No evaluation image or scene of
NYUv2, iBims-1 or ETH3D is in any pool.

Grid cells: `N400_k1 N400_k4 N400_k16 N1600_k1 N1600_k4 N1600_k16 N6400_k1 N6400_k4`, nested (image order fixed,
pixel indices 0..k−1 shared). Student: Qwen2.5-VL-3B-Instruct + LoRA r16 α32 on q/k/v/o, AdamW 1e-4 cosine,
batch 1 × accumulation 8, 2 epochs, seed fixed. Metric: δ1 (max(p/g, g/p) < 1.25) and AbsRel, with midpoint decoding.

### Evaluation sets

| Domain | Set | Used for | Images / pixels | Relation to the pools | Ground truth |
|---|---|---|---|---|---|
| Indoor | NYUv2 | indoor and mixed pools | 200 / 2,000 | same dataset, rooms excluded from the pools (pool v4) | Kinect |
| Indoor | iBims-1 | indoor and mixed pools | 100 / 3,000 | not in any pool | laser scan |
| Driving | DDAD | driving and mixed pools | 250 / 2,500 | different vehicle and countries, daytime; not in any pool | LiDAR |
| Driving | nuScenes | driving and mixed pools | 250 / 2,500 | DepthLM's evaluation split: 43 scenes of one evening in Singapore, all at night, 13 in rain; not in any pool | LiDAR |

Ground truth is the Euclidean distance from the camera centre to the point, the quantity the prompt asks for, in every
set. DepthLM's DDAD and nuScenes scripts use z-depth instead, so both driving sets also store z-depth, used only to
compare the teacher with the paper (DDAD δ1 0.670, nuScenes 0.819). ETH3D is no longer evaluated, since no pool is
judged on it.

DDAD and nuScenes are both in DepthLM's result tables, next to Qwen2.5-VL-3B without training, DepthLM-3B (the same
backbone trained on 16M ground-truth images) and pure vision models, so the driving columns can be set beside published
numbers. Both sets follow DepthLM's own curation scripts and subsample them to 250 images with 10 pixels each (seed 0),
keeping a 10-pixel border free so that the marker can be drawn. Our numbers therefore estimate the same quantity as the
paper's on other random pixels.

DDAD (`experiments/41_build_ddad.py`) follows `curate_ddad.py`: the 50 validation scenes, all six cameras, LiDAR
projected into the image with the nearest point kept per pixel, and pixels drawn at random from those with depth;
DepthLM uses every sample and 100 pixels per image, here five (sample, camera) pairs are drawn per scene. The procedure
does not mask the ego vehicle, and the five non-front cameras see part of it: of the 120 sampled pixels in the lower
quarter of those cameras, roughly half lie on the hood or on body panels reflecting the scene (visual check), and they
carry the depth of the ground behind the car. The DDAD column keeps these pixels to stay comparable with the published
numbers, and a sensitivity value without the lower quarter of the non-front cameras is reported beside it.

nuScenes (`experiments/43_build_nuscenes.py`) follows `curate_nuscenes_eval.py`: the last 5 % of the trainval sample list
(1,708 samples, the only nuScenes scenes the teacher was not trained on) with all six cameras, the top LiDAR projected with
the sensor calibrations only (no ego-pose correction, as in DepthLM), the nearest point per pixel, and pixels drawn at
random from those with depth; here 250 of the 10,248 (sample, camera) pairs are drawn. Because the sample list is ordered
by scene, this split is 43 consecutive scenes recorded on one evening, all at night and 13 in rain, so the nuScenes
column measures night-time driving. As in the official procedure, points closer than 1 m are not removed: three of the
2,500 pixels carry depths of 0.1 to 0.2 m from returns off the ego vehicle.

Measured on an RTX PRO 4500: 0.46 s per training step, ≈10 GB VRAM per cell; teacher labeling 0.8–1.5 s per pixel,
≈30 GB VRAM. H200 timings are to be measured.

## Local run

```bash
bash run.sh smoke                                   # installs requirements itself; no data needed
DATA_ROOT=/path/to/data bash run.sh grid mixed soft  # train and evaluate one grid
DATA_ROOT=/path/to/data OUT_ROOT=/path/with/checkpoints bash run.sh eval outdoor soft   # evaluate adapters from an H200 zip
```

`DATA_ROOT` must contain `pool/` and `eval/` (or the tar parts); the driving sets are built locally with
`experiments/41_build_ddad.py` and `experiments/43_build_nuscenes.py` from the original datasets and placed under
`$DATA_ROOT/eval/ddad` and `$DATA_ROOT/eval/nuscenes`. Results go to `./results` unless `OUT_ROOT` is set.

## License and attribution

MIT (`LICENSE`) covers only the code written for this repository. It does not cover the teacher pseudo-labels,
any checkpoint trained from them, the evaluation reference files or the vendored DepthLM code:

- `pools/*/teacher_labels.parquet` and trained adapters are outputs of the DepthLM model and are usable for
  noncommercial research only (FAIR Noncommercial Research License, copy in `third_party/DepthLM_Official/MODEL_LICENSE`).
  Publications must acknowledge DepthLM.
- `third_party/DepthLM_Official/utils/` is vendored from the DepthLM code repository under CC BY-NC 4.0.
- `ref/` contains sparse pixel coordinates and depth values from iBims-1, NYU Depth v2 and ETH3D under those datasets'
  research-use terms. `smoke/data/` images are synthetic. Full datasets are not redistributed.
- The DDAD and nuScenes evaluation sets are not in the repository. `experiments/41_build_ddad.py` and
  `experiments/43_build_nuscenes.py` rebuild them from the original datasets (both CC BY-NC-SA 4.0).

See `NOTICE` for details.
