#!/usr/bin/env bash
# DepthLM 증류 실험 — 컨테이너 진입점 (비대화형). 사용법:
#   MODE=smoke                       bash run.sh        # 파이프라인 검증 (모델 다운로드 → 30 스텝 학습 → 3 px 평가)
#   MODE=grid POOL=mixed COND=soft   bash run.sh        # 격자 8셀 학습 → 평가(small+large) → 표·그림   (COND = soft | hard)
# 환경변수: DATA_ROOT(데이터 루트, 기본 ./data), OUT_ROOT(결과, 기본 ./results), FOCAL(750), CELLS(기본 arms.json 전체), EVAL_SETS("small large"),
#           HF_TOKEN(gated 모델용), NPROC(병렬 학습 프로세스 수, MIG 슬라이스 격리 시 CUDA_VISIBLE_DEVICES 로 분배)
set -euo pipefail; cd "$(dirname "$0")"; export PYTHONUNBUFFERED=1
MODE=${MODE:-smoke}; POOL=${POOL:-mixed}; COND=${COND:-soft}; FOCAL=${FOCAL:-750}; EVAL_SETS=${EVAL_SETS:-"small large"}
export DATA_ROOT=${DATA_ROOT:-$PWD/data}; export OUT_ROOT=${OUT_ROOT:-$PWD/results}; mkdir -p "$OUT_ROOT"
LOG=$OUT_ROOT/run_${MODE}_${POOL}_${COND}.log; say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
say "MODE=$MODE POOL=$POOL COND=$COND FOCAL=$FOCAL DATA_ROOT=$DATA_ROOT OUT_ROOT=$OUT_ROOT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | tee -a "$LOG" || say "nvidia-smi 없음"
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a "$LOG"
if [ "$MODE" = smoke ]; then
  export DATA_ROOT=$PWD/smoke/data
  say "[smoke] 학습 30 스텝 (Qwen2.5-VL-3B 다운로드 포함)"
  python -u experiments/20_train_student.py --cond soft --steps 30 --accum 8 --focal "$FOCAL" --labels pools/smoke/teacher_labels.parquet --pools configs/pool_smoke.yaml --rows pools/smoke/rows_smoke.parquet --tag _smoke 2>&1 | grep -vE "^\[transformers\]" | tee -a "$LOG"
  say "[smoke] 평가 3 px (데이터셋당 1 이미지 1 픽셀)"
  python -u experiments/21_eval_student.py --tag smoke --adapter "$OUT_ROOT/checkpoints/soft_smoke" --focal "$FOCAL" --eval_set small --limit_img 1 --per_max 1 2>&1 | grep -vE "^\[transformers\]" | tee -a "$LOG"
  say "[smoke] 완료. 결과: $OUT_ROOT/eval/eval_smoke.parquet, 어댑터: $OUT_ROOT/checkpoints/soft_smoke"; exit 0
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
NPROC=${NPROC:-1}
if [ "$NPROC" -gt 1 ]; then   # MIG 슬라이스 격리 환경: 셀을 NPROC 개 프로세스에 라운드로빈 (각 프로세스는 자기 GPU 인덱스 사용)
  i=0; for cell in $CELLS; do CUDA_VISIBLE_DEVICES=$((i % NPROC)) train_cell "$cell" & i=$((i+1)); [ $((i % NPROC)) -eq 0 ] && wait; done; wait
else for cell in $CELLS; do train_cell "$cell"; done; fi
for cell in $CELLS; do ad=$OUT_ROOT/checkpoints/${COND}_${cell}_f${FOCAL}; [ -f "$ad/adapter_model.safetensors" ] || continue
  for es in $EVAL_SETS; do suf=""; [ "$es" = large ] && suf=_large; [ -f "$OUT_ROOT/eval/eval_${COND}_${cell}_f${FOCAL}${suf}.parquet" ] && continue
    say "평가 $COND $cell ($es)"; python -u experiments/21_eval_student.py --tag "${COND}_${cell}_f${FOCAL}" --adapter "$ad" --focal "$FOCAL" --eval_set "$es" > "$OUT_ROOT/eval_${COND}_${cell}_${es}.log" 2>&1 || say "!!! 평가 실패 $cell $es"; done; done
for es in $EVAL_SETS; do python experiments/31_grid.py --cond "$COND" --suffix "_f${FOCAL}" --eval_set "$es" --arms "$ARMS" >> "$LOG" 2>&1 || say "!!! 표 실패 $es"; done
say "[grid] 완료. 결과: $OUT_ROOT/{eval,tables,figures,checkpoints}"
