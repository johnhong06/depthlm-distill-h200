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

Submit one stage per request. On a service whose result volume survives between jobs, teacher inference, student
training and evaluation can each be their own job. On the 2026-09 service they do not, so teacher inference is one job
and training plus evaluation is a single `grid` job of about 21 hours.

| Issue | Command | GPU | What it does |
|---|---|---|---|
| 0 check | `bash run.sh check` | smallest slice | Reports, for `/app/data`, `/app/output` and the work root, whether it is writable and how much space is free and used, then where the archive would be extracted and whether anything that is not a result is sitting in `/app/output`. No GPU, no download, under a minute. **Run this first.** |
| 1 data | `bash run.sh data` | smallest slice | One-off check that the sixteen parts verify against `SHA256SUMS_parts` and unpack. It cannot hand the result to a later job on this service, because the work volume is emptied between jobs, so run it once and never again. A few minutes. |
| 2 smoke | `bash run.sh smoke` | 1 (18 GB slice) | Installs missing packages (and torch if too old), downloads the student, trains 30 steps on the bundled synthetic 40-image pool, evaluates 3 pixels. ~20 min. |
| 3 label | `bash run.sh label <mixed\|indoor\|outdoor> hf_xxx` | **7** (whole GPU) | Teacher inference. The 12B teacher labels only the pixels the repository labels do not already cover, with 4 concurrent processes. Writes `labels/<pool>/teacher_labels.parquet` and `labels_<pool>.zip`. Needs a token only if the teacher weights are not in the archive. |
| 4 train | `bash run.sh train <pool> <soft\|hard>` | **7** (whole GPU) | Student training, 8 cells concurrently. Reads the teacher labels and the pool images; the teacher weights are not loaded. Writes `checkpoints/`. ~12 h. No token needed. |
| 5 eval | `bash run.sh eval <pool> <soft\|hard>` | **7** (whole GPU) | Evaluates the 8 adapters on both evaluation sets, then writes tables, figures and the result zip. ~9 h. Reads neither the teacher labels nor the pool images. |

### How the teacher labels reach the training job

Teacher inference is the expensive stage (44,800 pixels per pool from scratch), so its output has to survive until the
training job runs. It is written to three places and training looks in all three, in this order:

1. `/app/output/labels/<pool>/teacher_labels.parquet`, the result proper (23 MB for a full pool).
2. `/app/data/labels/<pool>/teacher_labels.parquet`, a copy kept next to the data whenever `/app/data` is writable.
   This is what makes the split work if `/app/output` is not carried into the next pod.
3. `pools/<pool>/teacher_labels.parquet` in the repository, which every pod has by definition.

If neither `/app/output` nor `/app/data` survives between jobs, download `labels_<pool>.zip` after the labeling job and
commit the parquet to `pools/<pool>/teacher_labels.parquet`. The labeling job prints this instruction and the file size.
`train` refuses to start when it finds no labels and names all three paths it looked in; it also refuses when the labels
cover fewer pixels than the grid needs, and re-running `label` continues from what is already there instead of
starting over.

Notes.

- Re-submitting the same command resumes. A trained cell, a finished evaluation and a completed label shard are all
  skipped, so a job that was cut short only does the part that is missing. This works as long as `/app/output` is
  kept; clearing it starts every stage from scratch.
- `train` refuses to start when a pixel lacks a label and names the pool to label first. `eval` refuses to start when
  no adapter exists and names the training command.
- `grid <pool> <cond>` still runs train and eval in one job, and `all` still runs the whole chain. Neither is
  recommended on a shared pod: the 2026-09 `all` run was terminated after 35 hours and the log did not say how far
  it had got.
- On a whole GPU the script trains and evaluates 8 cells concurrently and labels with 4 teacher processes
  (≈30 GB each). On an 18 GB slice everything runs sequentially. Override with `NPROC`, `NPROC_LABEL`.
- Stdout is a summary only (the issue report is capped at 65,000 characters); full logs go to `/app/output`.

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

Ask the administrator for the six zip files and the two label files when the last evaluation finishes.

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

## Results

Every cell is `δ1 / AbsRel` on the large evaluation set, filled in as each cell's evaluation finishes, and a dash means
that cell has not been evaluated yet. The two numbers come from the cell's log line in `eval_<cond>_<cell>_<pool>_f750_large.log`,
which computes them exactly as `tables/table_grid_<cond>_<pool>_f750_large.md` does; the table adds the confidence intervals.

Each pool is judged on the evaluation sets of its own domain: the indoor pool on iBims-1 and NYUv2, the driving pool on
KITTI-HO and DDAD, and the mixed pool on all four. The other domain's columns are reported as transfer and are not used to
decide between cells. ETH3D (indoor and non-driving outdoor scenes) is reported for every pool as a reference. KITTI-HO
and DDAD were added on 2026-09-29 and are evaluated locally from the adapters in each result zip, so their columns fill
in later than the others. See [Evaluation sets](#evaluation-sets).

`soft` and `hard` are the two distillation losses, compared on identical pixels, labels, seeds and step counts. The last
two rows of each table are baselines rather than cells of the design. They are single models evaluated once, so they
carry no loss condition and the same value appears in all three pool tables.

### Progress

| Pool | Condition | Machine | Teacher labels | Training | Evaluation | Driving sets | Result file |
|---|---|---|---|---|---|---|---|
| mixed | soft | local | committed | done | 4/8 (large only) | pending | |
| mixed | hard | local | committed | | | pending | |
| indoor | soft | H200 | committed | | | pending | |
| indoor | hard | H200 | committed | | | pending | |
| outdoor | soft | H200 | v5: 5,640 px to add, do not submit yet | | | pending | |
| outdoor | hard | H200 | v5: 5,640 px to add, do not submit yet | | | pending | |

### Mixed pool (indoor 50 / driving 50)

| N | k | Budget | Loss | iBims-1 | NYUv2 | KITTI-HO | DDAD | ETH3D |
|---:|---:|---:|:--|:--|:--|:--|:--|:--|
| 400 | 1 | 400 | soft | 0.376 / 0.361 | 0.412 / 0.309 | — | — | 0.288 / 0.698 |
| | | | hard | — | — | — | — | — |
| 400 | 4 | 1600 | soft | 0.325 / 0.349 | 0.362 / 0.313 | — | — | 0.294 / 0.683 |
| | | | hard | — | — | — | — | — |
| 400 | 16 | 6400 | soft | 0.384 / 0.352 | 0.441 / 0.306 | — | — | 0.311 / 0.599 |
| | | | hard | — | — | — | — | — |
| 1600 | 1 | 1600 | soft | 0.352 / 0.350 | 0.431 / 0.307 | — | — | 0.315 / 0.586 |
| | | | hard | — | — | — | — | — |
| 1600 | 4 | 6400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 1600 | 16 | 25600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 6400 | 1 | 6400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 6400 | 4 | 25600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| **teacher, DepthLM 12B** | | | | 0.811 / 0.141 | 0.889 / 0.122 | — | — | 0.649 / 0.205 |
| **student, no distillation** | | | | — | — | — | — | — |

### Indoor pool

| N | k | Budget | Loss | iBims-1 | NYUv2 | KITTI-HO | DDAD | ETH3D |
|---:|---:|---:|:--|:--|:--|:--|:--|:--|
| 400 | 1 | 400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 400 | 4 | 1600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 400 | 16 | 6400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 1600 | 1 | 1600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 1600 | 4 | 6400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 1600 | 16 | 25600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 6400 | 1 | 6400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 6400 | 4 | 25600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| **teacher, DepthLM 12B** | | | | 0.811 / 0.141 | 0.889 / 0.122 | — | — | 0.649 / 0.205 |
| **student, no distillation** | | | | — | — | — | — | — |

### Driving pool

| N | k | Budget | Loss | iBims-1 | NYUv2 | KITTI-HO | DDAD | ETH3D |
|---:|---:|---:|:--|:--|:--|:--|:--|:--|
| 400 | 1 | 400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 400 | 4 | 1600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 400 | 16 | 6400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 1600 | 1 | 1600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 1600 | 4 | 6400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 1600 | 16 | 25600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 6400 | 1 | 6400 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| 6400 | 4 | 25600 | soft | — | — | — | — | — |
| | | | hard | — | — | — | — | — |
| **teacher, DepthLM 12B** | | | | 0.811 / 0.141 | 0.889 / 0.122 | — | — | 0.649 / 0.205 |
| **student, no distillation** | | | | — | — | — | — | — |

### Baselines

The teacher is `facebook/DepthLM` (12B) scored on exactly the pixels the students are scored on, with the same δ1 and
AbsRel definitions and the same image-cluster bootstrap as `31_grid.py`, so it is directly comparable. Its midpoint
correction is +0.005 because the teacher answers with two decimals, while a student answers with one and gets +0.05.
The numbers come from `ref/dist_<dataset>.parquet`, produced when the evaluation pixels were curated.

| Baseline | iBims-1 | NYUv2 | KITTI-HO | DDAD | ETH3D |
|---|:--|:--|:--|:--|:--|
| Teacher, DepthLM 12B, large set | 0.811 / 0.141 | 0.889 / 0.122 | — | — | 0.649 / 0.205 |
| Teacher, DepthLM 12B, small set | 0.793 / 0.156 | 0.875 / 0.124 | n/a | n/a | 0.657 / 0.183 |
| Student before distillation, Qwen2.5-VL-3B | — | — | — | — | — |

95% image-cluster bootstrap intervals for the teacher on the large set: iBims-1 [0.763, 0.856], NYUv2 [0.866, 0.911],
ETH3D [0.618, 0.678]. ETH3D is the lowest of the three for the teacher as well, so it bounds what a student can reach
on the held-out set.

The zero-shot student row needs one evaluation run with no adapter, which `21_eval_student.py` does when `--adapter` is
left empty. It is the baseline that says how much the distillation added.

Per-cell confidence intervals, the paired fixed-budget comparisons and the small-set tables are not repeated here. They
are in `tables/table_grid_<cond>_<pool>_f750[_large].md` inside each `results_<cond>_<pool>.zip`.

## Experiments

| Pool | Images / scenes | Domain | Labels |
|---|---|---|---|
| `mixed` | 6,400 / 3,684 | indoor 50 %, driving 50 % (matches DepthLM's per-dataset sampling) | 43,640 px included; 1,160 px (146 images replaced in v4) produced on the service |
| `indoor` | 6,400 / 5,674 | SUN RGB-D, NYUv2 | produced on the service |
| `outdoor` | 6,400 / 509 | KITTI (scene count saturates at 509; upper N cells add frames of the same drives, reported as a limitation) | produced on the service |

The driving pool's 509 scenes overstate the number of distinct places. The scene count treats each KITTI 2012 and KITTI
2015 image as its own scene, but these images were cut from raw drives, and 249 of the 788 in the outdoor pool (165 of
KITTI 2015, 84 of KITTI 2012) show the same place as a raw or depth-selection frame that is also in the pool (dHash
distance ≤ 10, correlation > 0.9); in the mixed pool it is 176 of 788. The campus drives add more repetition, since
they film one place under several drive numbers. This does not leak into any evaluation set, but it means the image
axis N adds fewer new places in the driving pool than the scene count suggests, and it is reported as a limitation
when N and k are compared there.

Pool v5 (2026-09-29, outdoor pool only): near-duplicate images are moved to the end of the image order (`experiments/07_dedup_pool_order.py`, report in `pools/outdoor/dedup_report.md`). Walking the order, an image goes to the back when an image already kept is within dHash distance 10 and above 0.9 grey-level correlation at 32×32. DepthLM subsampled "highly similar video frames" because they did not help, without stating a rule; this is an explicit version of that step. It moves 1,365 of the 6,400 outdoor images (893 raw frames, most of them taken while the car was stopped or slow, 195 KITTI 2015 and 146 KITTI 2012 images, which the pool had taken as `_10`/`_11` pairs 0.1 s apart, and 131 depth-selection frames). Distinct images among the first N go from 372 / 1,130 / 5,035 to 400 / 1,600 / 5,035 for N = 400 / 1,600 / 6,400, so the many-images cells are no longer handicapped by near-copies. The image set, the pixel coordinates and the N = 6,400 cells are unchanged; 5,640 pixels that moved into the first 1,600 images need new teacher labels, which are being produced locally. **Do not submit the outdoor grids until those labels are committed**; `run.sh` refuses to train on missing labels anyway. The same rule finds no near-duplicate in the indoor pool. The mixed pool (13 % near-duplicates at N = 1,600) keeps its current order for now, because its soft grid has already been trained.

Pool v4 (2026-09-23): NYUv2 has ~3 images per room, so the 200 NYUv2 evaluation images' 195 rooms are excluded
from the pools at the scene level (146 mixed-pool and 333 indoor-pool images replaced in place by the next unused
indoor candidates of the same deterministic order; see `pools/*/pool_v4_report.md`). No evaluation image or scene of
NYUv2, iBims-1 or ETH3D is in any pool.

Grid cells: `N400_k1 N400_k4 N400_k16 N1600_k1 N1600_k4 N1600_k16 N6400_k1 N6400_k4`, nested (image order fixed,
pixel indices 0..k−1 shared). Student: Qwen2.5-VL-3B-Instruct + LoRA r16 α32 on q/k/v/o, AdamW 1e-4 cosine,
batch 1 × accumulation 8, 2 epochs, seed fixed. Evaluation: `small` = 300 / 320 / 302 pixels
(iBims-1 / NYUv2 / ETH3D), `large` = 3,000 / 2,000 / 4,503 pixels. Metric: δ1 (max(p/g, g/p) < 1.25).

### Evaluation sets

| Domain | Set | Role | Images / pixels | Relation to the pools | Ground truth |
|---|---|---|---|---|---|
| Indoor | NYUv2 | near | 200 / 2,000 | same dataset, rooms excluded from the pools (pool v4) | Kinect |
| Indoor | iBims-1 | far | 100 / 3,000 | not in any pool | laser scan |
| Driving | KITTI-HO | near | 200 / 2,000 | same sensor and city; drives no pool uses, with every place that appears in a pool removed | accumulated LiDAR (KITTI annotated depth) |
| Driving | DDAD | far | 250 / 2,500 (in preparation) | different vehicle and countries, not in any pool | LiDAR |
| Mixed | ETH3D | reference | 454 / 4,503 | not in any pool | laser scan |

Ground truth is the Euclidean distance from the camera centre to the point, the quantity the prompt asks for, in every
set. DepthLM's own DDAD, nuScenes and Waymo scripts use z-depth instead, so KITTI-HO and DDAD also store z-depth, used
only to compare the teacher with the DDAD value in the DepthLM paper (δ1 0.670).

KITTI-HO (`experiments/40_build_kitti_heldout.py`) is held out by place, not only by drive number, because two kinds of
overlap survive a drive-level split:

- The campus drives (`2011_09_28`) film the same place under different drive numbers. A first draft that took one image
  from each of 50 unused campus drives had 19 of the 50 match a pool image (32×32 grey-level correlation > 0.9), so all
  81 unused campus drives are dropped.
- KITTI 2012 images, which the pools contain, were cut from raw drives, but KITTI 2012 publishes no drive mapping. Every
  fifth frame of each remaining drive is therefore compared with every KITTI image in the pools (dHash distance ≤ 10,
  then correlation > 0.9), and the 50 frames (5 s) on either side of a match are removed.

What remains are the 12 city, residential and road drives that no pool uses and that are not the source of a KITTI 2015
training image. 200 images are taken evenly spaced over them, at most one per second of drive (seed 0), with 10 pixels
per image where 0 < z ≤ 80 m, away from a 10-pixel border. `experiments/42_leak_check.py` repeats the comparison on the
finished set as an independent check. On the finished set it finds 232 pairs within dHash distance 6 and none above correlation
0.9 (highest 0.78, a different place on inspection). With only 12 drives, intervals on KITTI-HO are also reported with
the drive rather than the image as the bootstrap cluster.

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
