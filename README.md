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
| indoor | hard | H200 | v4 = v5 | committed | running | on H200 |
| outdoor | soft | H200 | v5 | committed (5,640 px added locally) | after indoor hard | on H200 (needs the driving pack) |
| outdoor | hard | H200 | v5 | committed (5,640 px added locally) | | on H200 (needs the driving pack) |

Order of work. On H200, one job at a time on a whole GPU: `grid indoor soft` (done) → `grid indoor hard` (running) →
`grid outdoor soft` → `grid outdoor hard`, the outdoor ones evaluated on H200 once the driving pack is in `/app/data`;
after each job, ask the administrator for its `results_<cond>_<pool>.zip`, which must contain `checkpoints/`. Locally:
the teacher on nuScenes mini, then `grid mixed soft` and `grid mixed hard` on pool v5 cell by cell, then the zero-shot
student on all four sets.

Pool v5 moves near-duplicate images to the end of the image order ([Pools](#pools)). The indoor pool has none, so its v4
and v5 are the same. A first mixed soft grid trained on v4, before the rule existed, is kept locally as a pilot and not
reported. The driving sets reach H200 as a separate pack (see [Running](#running)).

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

| Budget | N | k | Loss | iBims-1 | NYUv2 | DDAD | nuScenes |
|---:|---:|---:|:--|:--|:--|:--|:--|
| **400** | 400 | 1 | soft | 0.344 / 0.363 | 0.366 / 0.313 | 0.163 / 0.619 | — |
| | | | hard | 0.338 / 0.360 | 0.393 / 0.307 | 0.159 / 0.646 | — |
| **1,600** | 400 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
|  | 1600 | 1 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| **6,400** | 400 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
|  | 1600 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
|  | 6400 | 1 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| **25,600** | 1600 | 16 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
|  | 6400 | 4 | soft | — | — | — | — |
| | | | hard | — | — | — | — |
| **teacher, DepthLM 12B** | | | | 0.811 / 0.141 | 0.889 / 0.122 | 0.652 / 0.240 | — |
| **student, no distillation** | | | | — | — | — | — |

### Indoor pool

| Budget | N | k | Loss | iBims-1 | NYUv2 |
|---:|---:|---:|:--|:--|:--|
| **400** | 400 | 1 | soft | 0.322 / 0.364 | 0.403 / 0.307 |
| | | | hard | — | — |
| **1,600** | 400 | 4 | soft | 0.467 / 0.293 | 0.564 / 0.246 |
| | | | hard | — | — |
|  | 1600 | 1 | soft | 0.461 / 0.297 | 0.539 / 0.252 |
| | | | hard | — | — |
| **6,400** | 400 | 16 | soft | 0.600 / 0.217 | 0.703 / 0.189 |
| | | | hard | — | — |
|  | 1600 | 4 | soft | 0.596 / 0.218 | 0.730 / 0.180 |
| | | | hard | — | — |
|  | 6400 | 1 | soft | 0.602 / 0.216 | 0.724 / 0.181 |
| | | | hard | — | — |
| **25,600** | 1600 | 16 | soft | 0.666 / 0.199 | 0.777 / 0.169 |
| | | | hard | — | — |
|  | 6400 | 4 | soft | 0.664 / 0.199 | 0.773 / 0.160 |
| | | | hard | — | — |
| **teacher, DepthLM 12B** | | | | 0.811 / 0.141 | 0.889 / 0.122 |
| **student, no distillation** | | | | — | — |

### Driving pool

| Budget | N | k | Loss | DDAD | nuScenes |
|---:|---:|---:|:--|:--|:--|
| **400** | 400 | 1 | soft | — | — |
| | | | hard | — | — |
| **1,600** | 400 | 4 | soft | — | — |
| | | | hard | — | — |
|  | 1600 | 1 | soft | — | — |
| | | | hard | — | — |
| **6,400** | 400 | 16 | soft | — | — |
| | | | hard | — | — |
|  | 1600 | 4 | soft | — | — |
| | | | hard | — | — |
|  | 6400 | 1 | soft | — | — |
| | | | hard | — | — |
| **25,600** | 1600 | 16 | soft | — | — |
| | | | hard | — | — |
|  | 6400 | 4 | soft | — | — |
| | | | hard | — | — |
| **teacher, DepthLM 12B** | | | | 0.652 / 0.240 | — |
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
| Teacher, DepthLM 12B, measured | 0.811 / 0.141 | 0.889 / 0.122 | 0.652 / 0.240 (z: 0.677) | — |
| Student before distillation, Qwen2.5-VL-3B, measured | — | — | — | — |
| *paper:* DepthLM 12B (Pixtral) | 0.870 | 0.799 | 0.670 | 0.819 |
| *paper:* DepthLM 3B (Qwen2.5-VL-3B trained on 16M ground-truth images) | 0.890 | 0.868 | 0.724 | 0.870 |
| *paper:* Qwen2.5-VL-3B without training | 0.080 | 0.128 | 0.083 | 0.090 |

95 % image-cluster bootstrap intervals for the measured teacher: iBims-1 [0.763, 0.856], NYUv2 [0.866, 0.911]. On iBims-1
the measured teacher is 0.06 below the paper although inputs, preprocessing and decoding match the official code; on
ETH3D, which is no longer evaluated here, the same pipeline reproduced the paper within 0.009 (0.644 against 0.653).

The zero-shot student row needs one evaluation run with no adapter, which `21_eval_student.py` does when `--adapter` is
left empty. It is the baseline that says how much the distillation added.

## Running

How to use the Aerodrone H200 service itself (application, request form, what the container keeps, data delivery,
pitfalls) is on the [H200 guide page](https://johnhong06.github.io/depthlm-distill-h200/) (Korean).

| Command | GPU | What it does |
|---|---|---|
| `bash run.sh smoke` | 1 | 30 training steps and 3 evaluated pixels on the bundled synthetic pool, ~20 min |
| `bash run.sh label <pool>` | 7 | Teacher labels for the pixels the committed labels do not cover; commit the parquet to `pools/<pool>/` |
| `bash run.sh grid <pool> <soft\|hard>` | 7 | Trains the 8 cells, evaluates them on the pool's own sets and writes `results_<cond>_<pool>.zip` with `checkpoints/` |

- Data: the 30 GB archive (pool images, indoor evaluation sets, teacher weights) and the 566 MB driving pack
  `depthlm_drive_eval.tar` (+ `.sha256`) go anywhere under `/app/data`; `run.sh` finds, verifies and unpacks both.
- Nothing carries over between H200 jobs, so labels are committed to the repository and each grid trains and
  evaluates in one job.
- Locally, `DATA_ROOT` must contain `pool/` and `eval/` (or the tar parts), and results go to `OUT_ROOT`
  (default `./results`):

```bash
DATA_ROOT=/path/to/data bash run.sh grid mixed soft
DATA_ROOT=/path/to/data OUT_ROOT=/path/with/checkpoints bash run.sh eval outdoor soft   # adapters from an H200 zip
python experiments/32_decide.py --pool indoor --soft_root <OUT_ROOT> --hard_root <OUT_ROOT>   # decision table
```

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
| Driving | nuScenes | driving and mixed pools | 250 / 2,500 | DepthLM's actual evaluation split, nuScenes v1.0-mini: 10 scenes (7 day, 3 night) excluded from the teacher's training; not in any pool | LiDAR |

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

nuScenes (`experiments/43_build_nuscenes.py`) uses v1.0-mini, all 10 scenes (404 samples) and all six cameras, which is what the
paper evaluated: the released `curate_nuscenes_eval.py` takes the last 5 % of the trainval list instead, and the first
author confirmed in [issue #17](https://github.com/facebookresearch/DepthLM_Official/issues/17) that this was a mistake
in the released file, while the training script skips the mini scenes. We first built the released split (43
consecutive scenes, all at night) and measured the teacher at 0.640 against the paper's 0.819; no 10-scene subset of it
reaches the paper, which is how the mismatch was found. Projection follows the script (sensor calibrations only, nearest
point per pixel, points under 1 m kept); 250 of the 2,424 (sample, camera) pairs are drawn, 84 of them at night.

Measured on an RTX PRO 4500: 0.46 s per training step, ≈10 GB VRAM per cell; teacher labeling 0.8–1.5 s per pixel,
≈30 GB VRAM. H200 timings are to be measured.

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
