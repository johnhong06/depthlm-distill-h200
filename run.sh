#!/usr/bin/env bash
# DepthLM 증류 실험 — 컨테이너 진입점 (비대화형). 사용법:
#   MODE=smoke                       bash run.sh        # 파이프라인 검증 (모델 다운로드 → 30 스텝 학습 → 3 px 평가)
#   MODE=grid POOL=mixed COND=soft   bash run.sh        # 격자 8셀 학습 → 평가(small+large) → 표·그림   (COND = soft | hard)
# 환경변수: DATA_ROOT(데이터 루트, 기본 ./data), OUT_ROOT(결과, 기본 ./results), FOCAL(750), CELLS(기본 arms.json 전체), EVAL_SETS("small large"),
#           HF_TOKEN(gated 모델용), NPROC(병렬 학습 프로세스 수, MIG 슬라이스 격리 시 CUDA_VISIBLE_DEVICES 로 분배)
set -euo pipefail; cd "$(dirname "$0")"; export PYTHONUNBUFFERED=1
# 위치 인자: bash run.sh <smoke|label|grid|all> [pool] [cond] [hf_token]   (환경변수 MODE/POOL/COND/HF_TOKEN 도 동일하게 동작; 토큰은 /app/data/hf_token.txt 로도 가능)
# all = 혼합 격자 2개 → 실내·실외 라벨링(라벨이 없을 때만) → 실내·실외 격자 4개를 한 작업으로 이어서 실행
ARGS=(); for a in "$@"; do case $a in hf_*) export HF_TOKEN=$a;; *) ARGS+=("$a");; esac; done   # hf_ 로 시작하는 인자는 위치와 무관하게 토큰
MODE=${ARGS[0]:-${MODE:-smoke}}; POOL=${ARGS[1]:-${POOL:-mixed}}; COND=${ARGS[2]:-${COND:-soft}}
FOCAL=${FOCAL:-750}; EVAL_SETS=${EVAL_SETS:-"small large"}; export HF_HUB_DISABLE_PROGRESS_BARS=1
# 사업단 파드 규격: 데이터는 /app/data (읽기), 결과는 /app/output (파드 종료 후 보존). 없으면 로컬 기본값.
export DATA_ROOT=${DATA_ROOT:-$([ -d /app/data ] && echo /app/data || echo $PWD/data)}; export OUT_ROOT=${OUT_ROOT:-$([ -d /app/output ] && echo /app/output || echo $PWD/results)}; mkdir -p "$OUT_ROOT"
# 토큰: 기본은 이슈 명령 인자(hf_...). 대안으로 /app/data/hf_token.txt 파일도 읽는다
for tf in /app/data/hf_token.txt "$DATA_ROOT/hf_token.txt"; do [ -z "${HF_TOKEN:-}" ] && [ -f "$tf" ] && export HF_TOKEN=$(tr -d '[:space:]' < "$tf") && echo "[setup] HF token loaded from $tf"; done
# 파드(/app/output 존재)에서는 HF 가중치 캐시를 /app/output/hf 에 두어 다음 작업이 재다운로드하지 않게 한다
[ -d /app/output ] && export HF_HOME=${HF_HOME:-/app/output/hf}
# torch 가 2.5 미만이면(Docker Hub pytorch/pytorch:latest = 2.2.1) transformers 5 가 못 돌므로 cu128 빌드 2.11 로 교체 (호스트 드라이버 ≥ 570). 2.5 이상이면 손대지 않는다
python - <<'PYV' || { echo "[setup] torch 가 오래됨 → torch 2.11 + torchvision 0.26 (cu128) 설치, 3 GB"; pip uninstall -y -q torchaudio torchtext torchdata >/dev/null 2>&1 || true; pip install -q "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2; }   # 옛 torchaudio 는 새 torch 와 심볼이 안 맞아 import 를 깨뜨리므로 제거
import torch; v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); assert v >= (2, 5), torch.__version__
PYV
# 기본 이미지에 없는 모듈은 스스로 설치 (이슈에 "추가 모듈" 칸이 없어도 동작)
python - <<'PYV' || { echo "[setup] requirements 설치"; pip install -q -r requirements.txt 2>&1 | tail -2; }
import transformers, peft, accelerate, pandas, pyarrow, yaml, sklearn, matplotlib, tabulate, cv2; assert transformers.__version__ == "5.16.1", transformers.__version__
PYV
python -c "import torch, transformers, peft; print(f'[setup] torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__} cuda {torch.cuda.is_available()}')"
LOG=$OUT_ROOT/run_${MODE}_${POOL}_${COND}.log; say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
DATA_REPO=${DATA_REPO:-jh0624/depthlm-distill-data}
if [ -n "${HF_TOKEN:-}" ]; then python - "$DATA_REPO" <<'PYT' | tee -a "$LOG"
import sys; from huggingface_hub import HfApi
api = HfApi(); who = "?"
try: who = api.whoami()["name"]
except Exception as e: print(f"!!! [setup] 토큰 무효: {type(e).__name__}"); sys.exit(0)
for kind, rid in (("model", "facebook/DepthLM"), ("dataset", sys.argv[1])):
    try: (api.model_info if kind == "model" else api.dataset_info)(rid); print(f"[setup] 토큰({who}) → {rid} 접근 OK")
    except Exception as e: print(f"!!! [setup] 토큰({who}) → {rid} 접근 실패: {type(e).__name__} (라이선스 동의·비공개 저장소 권한 확인)")
PYT
else say "[setup] HF 토큰 없음 — 학생 모델(공개)만 가능. 라벨링·데이터 팩 다운로드는 토큰 필요"; fi
# 데이터 확보 순서: ① /app/data 에 풀려 있음 → ② 이전 작업이 /app/output/data 에 풀어 둠 → ③ /app/data 의 tar 분할본 → ④ HF 비공개 데이터셋(DATA_REPO)에서 토큰으로 내려받음
if [ ! -d "$DATA_ROOT/pool" ]; then
  if [ -d "$OUT_ROOT/data/pool" ]; then export DATA_ROOT=$OUT_ROOT/data
  else
    PACK=""; ls "$DATA_ROOT"/depthlm_distill_data.tar.part_* >/dev/null 2>&1 && PACK=$DATA_ROOT
    if [ -z "$PACK" ] && [ -n "${HF_TOKEN:-}" ]; then
      echo "[data] $DATA_REPO 에서 데이터 팩 다운로드 (6 GB)"; mkdir -p "$OUT_ROOT/data_pack"
      python - "$DATA_REPO" "$OUT_ROOT/data_pack" <<'PYD' && PACK=$OUT_ROOT/data_pack || echo "!!! [data] 다운로드 실패 — 토큰이 $DATA_REPO 를 읽을 수 있는지 확인"
import sys, time; from huggingface_hub import snapshot_download
repo, out = sys.argv[1:3]
for a in range(3):
    try: snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=["depthlm_distill_data.tar.part_*", "SHA256SUMS"]); print("[data] 다운로드 완료"); break
    except Exception as e:
        print(f"!!! [data] 시도 {a+1} 실패: {type(e).__name__}: {str(e)[:120]}")
        if type(e).__name__ in ("RepositoryNotFoundError", "GatedRepoError"): sys.exit(1)   # 권한·이름 문제는 재시도 무의미
        time.sleep(30)
else: sys.exit(1)
PYD
    fi
    if [ -n "$PACK" ]; then
      [ -f "$PACK/SHA256SUMS" ] && { (cd "$PACK" && sha256sum -c --quiet SHA256SUMS) && echo "[data] SHA256 검증 통과" || { echo "!!! [data] SHA256 불일치 — 분할본이 깨짐"; exit 1; }; }
      mkdir -p "$OUT_ROOT/data"; cat "$PACK"/depthlm_distill_data.tar.part_* | tar -xf - -C "$OUT_ROOT/data" --strip-components=1 && export DATA_ROOT=$OUT_ROOT/data
      [ "$PACK" = "$OUT_ROOT/data_pack" ] && rm -rf "$OUT_ROOT/data_pack"; echo "[data] 풀기 완료: $(find "$DATA_ROOT/pool" -type f | wc -l) 풀 이미지, $(find "$DATA_ROOT/eval" -type f | wc -l) 평가 파일"
    fi
  fi
fi
# 모델 가중치가 /app/data/models 에 있으면 그것을 쓰고, 없으면 Hugging Face 에서 내려받음 (인터넷 필요)
[ -d "$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct" ] && export STUDENT_MODEL=$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct
[ -d "$DATA_ROOT/models/DepthLM" ] && export TEACHER_MODEL=$DATA_ROOT/models/DepthLM
say "MODE=$MODE POOL=$POOL COND=$COND FOCAL=$FOCAL DATA_ROOT=$DATA_ROOT OUT_ROOT=$OUT_ROOT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | tee -a "$LOG" || say "nvidia-smi 없음"
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
[ -n "${GPU_MB:-}" ] || GPU_MB=$(python -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/2**20) if torch.cuda.is_available() else 0)")   # nvidia-smi 가 없는 이미지
DEVS=($(nvidia-smi -L 2>/dev/null | grep -oE "MIG-[0-9a-f-]+" || true)); NDEV=${#DEVS[@]}   # MIG 슬라이스가 여러 개 보이면 셀을 슬라이스별로 분배
NPROC_TRAIN=${NPROC:-$([ "${GPU_MB:-0}" -gt 100000 ] && echo 8 || { [ "$NDEV" -gt 1 ] && echo "$NDEV" || echo 1; })}; NPROC_LABEL=${NPROC_LABEL:-$([ "${GPU_MB:-0}" -gt 100000 ] && echo 4 || echo 1)}
say "GPU ${GPU_MB} MiB, MIG 장치 $NDEV → 학습 병렬 $NPROC_TRAIN, 라벨링 병렬 $NPROC_LABEL"
if [ "$MODE" = label ] || [ "$MODE" = all ]; then [ "${GPU_MB:-0}" -ge 28000 ] || say "!!! 장치 메모리 ${GPU_MB} MiB < 28 GB — 교사(12B)가 들어가지 않아 라벨링은 실패함. GPU 할당량 7(통째)로 요청할 것"; fi
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a "$LOG"
if [ "$MODE" = smoke ]; then
  export DATA_ROOT=$PWD/smoke/data
  say "[smoke] 학습 30 스텝 (Qwen2.5-VL-3B 다운로드 포함)"
  python -u experiments/20_train_student.py --cond soft --steps 30 --accum 8 --focal "$FOCAL" --labels pools/smoke/teacher_labels.parquet --pools configs/pool_smoke.yaml --rows pools/smoke/rows_smoke.parquet --tag _smoke 2>&1 | grep -vE "^\[transformers\]" | tee -a "$LOG"
  say "[smoke] 평가 3 px (데이터셋당 1 이미지 1 픽셀)"
  python -u experiments/21_eval_student.py --tag smoke --adapter "$OUT_ROOT/checkpoints/soft_smoke" --focal "$FOCAL" --eval_set small --limit_img 1 --per_max 1 2>&1 | grep -vE "^\[transformers\]" | tee -a "$LOG"
  say "[smoke] 완료. 결과: $OUT_ROOT/eval/eval_smoke.parquet, 어댑터: $OUT_ROOT/checkpoints/soft_smoke"; exit 0
fi
if [ "$MODE" = all ]; then   # 한 이슈로 전체 체인. 각 단계는 하위 실행이라 하나가 실패해도 다음으로 넘어간다
  for c in soft hard; do say "[all] 격자 mixed $c"; bash run.sh grid mixed $c || say "!!! [all] 격자 실패 mixed $c"; done
  for p in indoor outdoor; do
    if [ -f pools/$p/teacher_labels.parquet ] || [ -f "$OUT_ROOT/labels/$p/teacher_labels.parquet" ]; then say "[all] $p 라벨 있음 — 라벨링 건너뜀"
    else [ -n "${HF_TOKEN:-}" ] || say "!!! [all] HF_TOKEN 없음 — $p 라벨링은 실패할 것"; say "[all] 라벨링 $p"; bash run.sh label $p || say "!!! [all] 라벨링 실패 $p"; fi; done
  # 라벨링이 둘 다 끝났으면 토큰 사본을 지운다 (이후 격자는 토큰 불필요). /app/data 가 읽기 전용이면 관리자에게 삭제 요청. 실제 무효화는 HF 계정에서 Revoke 해야 한다
  if [ -f "$OUT_ROOT/labels/indoor/teacher_labels.parquet" ] && [ -f "$OUT_ROOT/labels/outdoor/teacher_labels.parquet" ]; then
    for tf in /app/data/hf_token.txt "$DATA_ROOT/hf_token.txt"; do [ -f "$tf" ] && { rm -f "$tf" 2>/dev/null && say "[all] 토큰 파일 삭제됨: $tf" || say "!!! [all] 토큰 파일을 지우지 못함(읽기 전용): $tf — 관리자에게 삭제 요청"; }; done
    unset HF_TOKEN; say "[all] 라벨링 완료 — 이후 단계는 토큰을 쓰지 않음. HF 설정에서 토큰을 Revoke 할 것"
  else say "!!! [all] 라벨이 둘 다 없어 토큰 파일을 남겨 둠(재실행용)"; fi
  for p in indoor outdoor; do for c in soft hard; do say "[all] 격자 $p $c"; bash run.sh grid $p $c || say "!!! [all] 격자 실패 $p $c"; done; done
  say "[all] 완료. 결과 zip: $(ls "$OUT_ROOT"/results_*.zip 2>/dev/null | tr '\n' ' ')"; exit 0
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
# --- grid ---  태그 = <cond>_<cell>_<pool>_f<focal>  (풀이 달라도 체크포인트·평가 파일이 겹치지 않음)
SUFFIX=_${POOL}_f${FOCAL}; ARMS=pools/$POOL/arms.json; PCFG=configs/pool_${POOL}.yaml
LABELS=pools/$POOL/teacher_labels.parquet; [ -f "$LABELS" ] || LABELS=$OUT_ROOT/labels/$POOL/teacher_labels.parquet   # 저장소에 없으면 파드에서 만든 라벨
[ -f "$ARMS" ] && [ -f "$LABELS" ] && [ -f "$PCFG" ] || { say "!!! 풀 파일 없음: $ARMS $LABELS $PCFG"; exit 1; }
[ -d "$DATA_ROOT/pool" ] && [ -d "$DATA_ROOT/eval" ] || { say "!!! DATA_ROOT 에 pool/ eval/ 없음 → scripts/fetch_data.sh 먼저"; exit 1; }
CELLS=${CELLS:-$(python -c "import json;print(' '.join(c['tag'] for c in json.load(open('$ARMS'))['cells']))")}
say "[grid] 풀 $POOL 조건 $COND 라벨 $LABELS 셀: $CELLS"
train_cell() { local cell=$1; local ad=$OUT_ROOT/checkpoints/${COND}_${cell}${SUFFIX}
  [ -f "$ad/adapter_model.safetensors" ] && { say "$cell 학습 완료됨 — 건너뜀"; return 0; }
  say "학습 $COND $cell"; python -u experiments/20_train_student.py --cond "$COND" --epochs 2 --accum 8 --focal "$FOCAL" --labels "$LABELS" --pools "$PCFG" --rows "pools/$POOL/rows_$cell.parquet" --tag "_${cell}${SUFFIX}" > "$OUT_ROOT/train_${COND}_${cell}${SUFFIX}.log" 2>&1 || say "!!! 학습 실패 $cell"; }
eval_cell() { local cell=$1; local ad=$OUT_ROOT/checkpoints/${COND}_${cell}${SUFFIX}; [ -f "$ad/adapter_model.safetensors" ] || return 0
  for es in $EVAL_SETS; do local suf=""; [ "$es" = large ] && suf=_large; [ -f "$OUT_ROOT/eval/eval_${COND}_${cell}${SUFFIX}${suf}.parquet" ] && continue
    say "평가 $COND $cell ($es)"; python -u experiments/21_eval_student.py --tag "${COND}_${cell}${SUFFIX}" --adapter "$ad" --focal "$FOCAL" --eval_set "$es" ${EVAL_EXTRA:-} > "$OUT_ROOT/eval_${COND}_${cell}${SUFFIX}_${es}.log" 2>&1 || say "!!! 평가 실패 $cell $es"; done; }
run_cells() { local fn=$1; if [ "$NPROC_TRAIN" -gt 1 ]; then   # GPU 한 장 통째: 셀 NPROC_TRAIN 개를 같은 GPU 에서 동시에 (학습 ≈10 GB, 평가 ≈8 GB)
    local i=0; for cell in $CELLS; do if [ "$NDEV" -gt 1 ]; then CUDA_VISIBLE_DEVICES=${DEVS[$((i % NDEV))]} $fn "$cell" & else $fn "$cell" & fi; i=$((i+1)); [ $((i % NPROC_TRAIN)) -eq 0 ] && wait; done; wait
  else for cell in $CELLS; do $fn "$cell"; done; fi; }
run_cells train_cell; run_cells eval_cell
for es in $EVAL_SETS; do python experiments/31_grid.py --cond "$COND" --suffix "$SUFFIX" --eval_set "$es" --arms "$ARMS" >> "$LOG" 2>&1 || say "!!! 표 실패 $es"; done
say "[grid] 완료 $POOL $COND. 결과: $OUT_ROOT/{eval,tables,figures,checkpoints}"
for es in $EVAL_SETS; do suf=""; [ "$es" = large ] && suf=_large; echo "===== 요약 $POOL $COND ($es) ====="; head -40 "$OUT_ROOT/tables/table_grid_${COND}${SUFFIX}${suf}.md" 2>/dev/null; done
python - "$OUT_ROOT" "results_${COND}_${POOL}" "${COND}_" "$SUFFIX" <<'PYS'
import sys, os, zipfile; root, name, cond, suffix = sys.argv[1:5]
with zipfile.ZipFile(f"{root}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("hf", "data", "labels")]
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), root)
            if (cond in rel and suffix in rel) or rel.startswith("run_"): z.write(os.path.join(dp, f), rel)
print(f"zip: {root}/{name}.zip  {os.path.getsize(f'{root}/{name}.zip')/1e6:.1f} MB")
PYS
