#!/usr/bin/env bash
# DepthLM 증류 실험 — 컨테이너 진입점 (비대화형). 사용법:
#   MODE=smoke                       bash run.sh        # 파이프라인 검증 (모델 다운로드 → 30 스텝 학습 → 3 px 평가)
#   MODE=grid POOL=mixed COND=soft   bash run.sh        # 격자 8셀 학습 → 평가(small+large) → 표·그림   (COND = soft | hard)
# 환경변수: DATA_ROOT(데이터 루트, 기본 ./data), OUT_ROOT(결과, 기본 ./results), FOCAL(750), CELLS(기본 arms.json 전체), EVAL_SETS("small large"),
#           HF_TOKEN(gated 모델용), NPROC(병렬 학습 프로세스 수, MIG 슬라이스 격리 시 CUDA_VISIBLE_DEVICES 로 분배)
set -euo pipefail; cd "$(dirname "$0")"; export PYTHONUNBUFFERED=1
# 위치 인자: bash run.sh <smoke|label|grid> [pool] [cond] [hf_token]   (환경변수 MODE/POOL/COND/HF_TOKEN 도 동일하게 동작)
MODE=${1:-${MODE:-smoke}}; POOL=${2:-${POOL:-mixed}}; COND=${3:-${COND:-soft}}; [ -n "${4:-}" ] && export HF_TOKEN=$4
FOCAL=${FOCAL:-750}; EVAL_SETS=${EVAL_SETS:-"small large"}; export HF_HUB_DISABLE_PROGRESS_BARS=1
# 사업단 파드 규격: 데이터는 /app/data (읽기), 결과는 /app/output (파드 종료 후 보존). 없으면 로컬 기본값.
export DATA_ROOT=${DATA_ROOT:-$([ -d /app/data ] && echo /app/data || echo $PWD/data)}; export OUT_ROOT=${OUT_ROOT:-$([ -d /app/output ] && echo /app/output || echo $PWD/results)}; mkdir -p "$OUT_ROOT"
# /app/data 에 tar 분할본만 있고 풀이 안 풀려 있으면 쓰기 가능한 곳에 풀어서 사용
if [ ! -d "$DATA_ROOT/pool" ] && ls "$DATA_ROOT"/depthlm_distill_data.tar.part_* >/dev/null 2>&1; then
  mkdir -p "$OUT_ROOT/data"; cat "$DATA_ROOT"/depthlm_distill_data.tar.part_* | tar -xf - -C "$OUT_ROOT/data" --strip-components=1; export DATA_ROOT=$OUT_ROOT/data; fi
# 모델 가중치가 /app/data/models 에 있으면 그것을 쓰고, 없으면 Hugging Face 에서 내려받음 (인터넷 필요)
[ -d "$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct" ] && export STUDENT_MODEL=$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct
[ -d "$DATA_ROOT/models/DepthLM" ] && export TEACHER_MODEL=$DATA_ROOT/models/DepthLM
LOG=$OUT_ROOT/run_${MODE}_${POOL}_${COND}.log; say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
say "MODE=$MODE POOL=$POOL COND=$COND FOCAL=$FOCAL DATA_ROOT=$DATA_ROOT OUT_ROOT=$OUT_ROOT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | tee -a "$LOG" || say "nvidia-smi 없음"
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
NPROC_TRAIN=${NPROC:-$([ "${GPU_MB:-0}" -gt 100000 ] && echo 8 || echo 1)}; NPROC_LABEL=${NPROC_LABEL:-$([ "${GPU_MB:-0}" -gt 100000 ] && echo 4 || echo 1)}
say "GPU ${GPU_MB} MiB → 학습 병렬 $NPROC_TRAIN, 라벨링 병렬 $NPROC_LABEL"
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a "$LOG"
if [ "$MODE" = smoke ]; then
  export DATA_ROOT=$PWD/smoke/data
  say "[smoke] 학습 30 스텝 (Qwen2.5-VL-3B 다운로드 포함)"
  python -u experiments/20_train_student.py --cond soft --steps 30 --accum 8 --focal "$FOCAL" --labels pools/smoke/teacher_labels.parquet --pools configs/pool_smoke.yaml --rows pools/smoke/rows_smoke.parquet --tag _smoke 2>&1 | grep -vE "^\[transformers\]" | tee -a "$LOG"
  say "[smoke] 평가 3 px (데이터셋당 1 이미지 1 픽셀)"
  python -u experiments/21_eval_student.py --tag smoke --adapter "$OUT_ROOT/checkpoints/soft_smoke" --focal "$FOCAL" --eval_set small --limit_img 1 --per_max 1 2>&1 | grep -vE "^\[transformers\]" | tee -a "$LOG"
  say "[smoke] 완료. 결과: $OUT_ROOT/eval/eval_smoke.parquet, 어댑터: $OUT_ROOT/checkpoints/soft_smoke"; exit 0
fi
if [ "$MODE" = label ]; then   # 교사 라벨링: VRAM 28-30 GB → MIG 7 작업. 결과 라벨은 $OUT_ROOT/labels/<POOL>/teacher_labels.parquet (재개 가능)
  [ -d "$DATA_ROOT/pool" ] || { say "!!! DATA_ROOT 에 pool/ 없음"; exit 1; }; mkdir -p "$OUT_ROOT/labels/$POOL"
  say "[label] 풀 $POOL: $(python -c "import pandas as pd;print(len(pd.read_parquet('pools/$POOL/todo_label.parquet')))") px, 병렬 $NPROC_LABEL"
  python - "$POOL" "$NPROC_LABEL" "$OUT_ROOT" <<'PYS'
import sys, os, pandas as pd; pool, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t = pd.read_parquet(f"pools/{pool}/todo_label.parquet"); os.makedirs(f"{out}/labels/{pool}", exist_ok=True)
for i in range(n): t.iloc[i::n].to_parquet(f"{out}/labels/{pool}/todo_{i}.parquet", index=False)
PYS
  for i in $(seq 0 $((NPROC_LABEL-1))); do
    ( for a in 1 2 3; do python -u experiments/11_label_teacher.py --pool pools/$POOL/pool.jsonl --image_folder "$DATA_ROOT" --todo "$OUT_ROOT/labels/$POOL/todo_$i.parquet" --out "$OUT_ROOT/labels/$POOL/part_$i.parquet" --chunk 8 > "$OUT_ROOT/labels/$POOL/label_$i.log" 2>&1 && break; sleep 30; done ) &
  done; wait
  python - "$POOL" "$OUT_ROOT" <<'PYS'
import sys, glob, pandas as pd; pool, out = sys.argv[1], sys.argv[2]
d = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{out}/labels/{pool}/part_*.parquet"))], ignore_index=True).drop_duplicates(["image_id", "pixel_index"])
d.to_parquet(f"{out}/labels/{pool}/teacher_labels.parquet", index=False); print(f"[label] 병합 {len(d)} px, 파싱 실패 {d.teacher_greedy1.isna().mean()*100:.2f}%, 질량 중앙 {d.teacher_mass.median():.3f}")
PYS
  say "[label] 완료: $OUT_ROOT/labels/$POOL/teacher_labels.parquet"; exit 0
fi
# --- grid ---
ARMS=pools/$POOL/arms.json; LABELS=pools/$POOL/teacher_labels.parquet; PCFG=configs/pool_${POOL}.yaml
[ -f "$ARMS" ] && [ -f "$LABELS" ] && [ -f "$PCFG" ] || { say "!!! 풀 파일 없음: $ARMS $LABELS $PCFG"; exit 1; }
[ -d "$DATA_ROOT/pool" ] && [ -d "$DATA_ROOT/eval" ] || { say "!!! DATA_ROOT 에 pool/ eval/ 없음 → scripts/fetch_data.sh 먼저"; exit 1; }
CELLS=${CELLS:-$(python -c "import json;print(' '.join(c['tag'] for c in json.load(open('$ARMS'))['cells']))")}
say "[grid] 셀: $CELLS"
train_cell() { local cell=$1; local ad=$OUT_ROOT/checkpoints/${COND}_${cell}_f${FOCAL}
  [ -f "$ad/adapter_model.safetensors" ] && { say "$cell 학습 완료됨 — 건너뜀"; return 0; }
  say "학습 $COND $cell"; python -u experiments/20_train_student.py --cond "$COND" --epochs 2 --accum 8 --focal "$FOCAL" --labels "$LABELS" --pools "$PCFG" --rows "pools/$POOL/rows_$cell.parquet" --tag "_${cell}_f${FOCAL}" > "$OUT_ROOT/train_${COND}_${cell}.log" 2>&1 || say "!!! 학습 실패 $cell"; }
if [ "$NPROC_TRAIN" -gt 1 ]; then   # GPU 한 장 통째: 셀 NPROC_TRAIN 개를 같은 GPU 에서 동시에 (셀당 ≈10 GB)
  i=0; for cell in $CELLS; do train_cell "$cell" & i=$((i+1)); [ $((i % NPROC_TRAIN)) -eq 0 ] && wait; done; wait
else for cell in $CELLS; do train_cell "$cell"; done; fi
for cell in $CELLS; do ad=$OUT_ROOT/checkpoints/${COND}_${cell}_f${FOCAL}; [ -f "$ad/adapter_model.safetensors" ] || continue
  for es in $EVAL_SETS; do suf=""; [ "$es" = large ] && suf=_large; [ -f "$OUT_ROOT/eval/eval_${COND}_${cell}_f${FOCAL}${suf}.parquet" ] && continue
    say "평가 $COND $cell ($es)"; python -u experiments/21_eval_student.py --tag "${COND}_${cell}_f${FOCAL}" --adapter "$ad" --focal "$FOCAL" --eval_set "$es" > "$OUT_ROOT/eval_${COND}_${cell}_${es}.log" 2>&1 || say "!!! 평가 실패 $cell $es"; done; done
for es in $EVAL_SETS; do python experiments/31_grid.py --cond "$COND" --suffix "_f${FOCAL}" --eval_set "$es" --arms "$ARMS" >> "$LOG" 2>&1 || say "!!! 표 실패 $es"; done
say "[grid] 완료. 결과: $OUT_ROOT/{eval,tables,figures,checkpoints}"
echo "===== 요약 (표) ====="; cat "$OUT_ROOT"/tables/table_grid_${COND}_f${FOCAL}.md 2>/dev/null | head -60
python - "$OUT_ROOT" "results_${COND}_${POOL}" <<'PYS'
import sys, os, zipfile, glob; root, name = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(f"{root}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in [q for d in ("eval", "tables", "figures", "checkpoints") for q in glob.glob(f"{root}/{d}/**", recursive=True) if os.path.isfile(q)] + glob.glob(f"{root}/*.log"): z.write(p, os.path.relpath(p, root))
print(f"zip: {root}/{name}.zip  {os.path.getsize(f'{root}/{name}.zip')/1e6:.1f} MB")
PYS
