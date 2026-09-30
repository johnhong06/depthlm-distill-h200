#!/usr/bin/env bash
# DepthLM 증류 실험 — 컨테이너 진입점 (비대화형). 사용법:
#   MODE=check                       bash run.sh        # 경로·쓰기 권한·여유 용량만 점검 (GPU 불필요, 1 분). 압축 해제 위치를 담당자와 확인할 때
#   MODE=data                        bash run.sh        # tar 조각 검증 + 압축 해제만 (GPU 불필요, 수 분). 한 번만 하면 이후 작업이 재사용한다
#   MODE=smoke                       bash run.sh        # 파이프라인 검증 (모델 다운로드 → 30 스텝 학습 → 3 px 평가)
# 권장: 세 단계를 따로 요청한다 (한 작업에 몰면 24 시간을 넘기고, 중간에 죽으면 어디까지 됐는지 알기 어렵다)
#   MODE=label POOL=mixed            bash run.sh        # ① 교사 추론 = 라벨링          → /app/output/labels/<pool>/teacher_labels.parquet
#   MODE=train POOL=mixed COND=soft  bash run.sh        # ② 학생 학습 8셀               → /app/output/checkpoints/
#   MODE=eval  POOL=mixed COND=soft  bash run.sh        # ③ 평가(small+large) → 표·그림·zip
#   MODE=grid  POOL=mixed COND=soft  bash run.sh        # ②+③ 을 한 작업에 (이전 방식, 호환용)
# 환경변수: DATA_ROOT(데이터 루트, 기본 ./data), OUT_ROOT(결과, 기본 ./results), FOCAL(750), CELLS(기본 arms.json 전체), EVAL_SETS(기본 large, none = 평가 안 함), EVAL_DATASETS(기본 풀별),
#           HF_TOKEN(gated 모델용), NPROC(병렬 학습 프로세스 수, MIG 슬라이스 격리 시 CUDA_VISIBLE_DEVICES 로 분배)
set -euo pipefail; cd "$(dirname "$0")"; export PYTHONUNBUFFERED=1
# 규칙: /app/output 은 결과 전용이다. 압축 해제본·모델 캐시·임시 파일은 절대 여기에 두지 않는다 (2026-09-27 output 100 GB 초과로 실험이 강제 종료된 원인)
writable() { [ -d "$1" ] || return 1; touch "$1/.h200_write_test" 2>/dev/null || return 1; rm -f "$1/.h200_write_test" 2>/dev/null || true; return 0; }
freegb() { local g; g=$(df -Pk "$1" 2>/dev/null | awk 'NR==2{printf "%d", int($4/1048576)}') || g=""; echo "${g:-?}"; }
usedgb() { local g; g=$(timeout 60 du -sx -k "$1" 2>/dev/null | tail -1 | awk '{printf "%d", int($1/1048576)}') || g=""; echo "${g:-?}"; }   # 공유 마운트가 커도 60 초에 끊는다
have_members() { local root=$1 m; for m in ${2:-pool}; do [ -e "$root/$m" ] || return 1; done; return 0; }   # 부분 해제본이 남아 있을 때 이 단계에 필요한 폴더가 다 있는지
NEED_GB_EXPLICIT=${NEED_GB:+1}; NEED_GB=${NEED_GB:-32}   # 30 GB 팩을 풀기 전에 요구하는 최소 여유 공간 (GB). 명시하면 단계별 기본값보다 우선한다
needgb() { local fg; fg=$(freegb "$1"); case $fg in ''|'?') echo "[data] $1 여유 공간 확인 불가 — 계속";; *) [ "$fg" -ge "$2" ] || { echo "!!! [data] $1 여유 ${fg} GB < 필요 $2 GB — 풀 공간이 없다. 관리자에게 여유 공간을 요청할 것"; exit 1; };; esac; }
EXTRACTED_HERE=0   # 이 작업이 $XROOT 에 데이터를 풀었으면 1
# 해제본은 /app/output 이 아니라 작업 볼륨에 있으므로 지울 이유가 없다. 30 GB 를 rm -rf 하느라 작업 끝에 시간을 쓰지 않는다.
# 2026-09-28 실측: /app/scratch 는 작업 사이에 유지되지 않으므로 파드가 끝나면 알아서 사라진다. 굳이 지우려면 CLEAN_WORK=1 을 준다
cleanup() { if [ "${CLEAN_WORK:-0}" = 1 ] && [ "${H200_CHILD:-0}" = 0 ] && [ "$EXTRACTED_HERE" = 1 ] && [ -n "${XROOT:-}" ]; then
    rm -rf "$XROOT" && echo "[cleanup] 해제본 삭제 ($XROOT)"; fi; return 0; }
trap cleanup EXIT
# 위치 인자: bash run.sh <smoke|label|grid|all> [pool] [cond] [hf_token]   (환경변수 MODE/POOL/COND/HF_TOKEN 도 동일하게 동작; 토큰은 /app/data/hf_token.txt 로도 가능)
# all = 혼합 격자 2개 → 실내·실외 라벨링(라벨이 없을 때만) → 실내·실외 격자 4개를 한 작업으로 이어서 실행
ARGS=(); for a in "$@"; do case $a in hf_*) export HF_TOKEN=$a;; *) ARGS+=("$a");; esac; done   # hf_ 로 시작하는 인자는 위치와 무관하게 토큰
MODE=${ARGS[0]:-${MODE:-smoke}}; POOL=${ARGS[1]:-${POOL:-mixed}}; COND=${ARGS[2]:-${COND:-soft}}
case $MODE in check|data|smoke|label|train|eval|grid|all) ;; *) echo "!!! 알 수 없는 MODE=$MODE — check|data|smoke|label|train|eval|grid|all 중 하나"; exit 1;; esac
FOCAL=${FOCAL:-750}; EVAL_SETS=${EVAL_SETS:-large}; export HF_HUB_DISABLE_PROGRESS_BARS=1   # small ⊂ large 이고 복호가 결정적이라 small 은 large 에서 골라낸다 (NOTES D-17)
# 사업단 파드 규격: 데이터는 /app/data, 결과는 /app/output (파드 종료 후 보존). 없으면 로컬 기본값.
export DATA_ROOT=${DATA_ROOT:-$([ -d /app/data/depthlm_distill_h200 ] && echo /app/data/depthlm_distill_h200 || { [ -d /app/data ] && echo /app/data || echo $PWD/data; })}   # 관리자가 tar 를 푼 폴더 우선
export OUT_ROOT=${OUT_ROOT:-$([ -d /app/output ] && echo /app/output || echo $PWD/results)}; mkdir -p "$OUT_ROOT"
DATA_SRC=$DATA_ROOT   # 팩 파일이 놓인 원래 위치 (풀린 뒤 DATA_ROOT 가 바뀌어도 추가 팩은 여기서 찾는다)
# 압축 해제본·모델 캐시가 갈 곳. /app/output 은 후보가 아니다 (결과 전용, /app/data 와 같은 볼륨이라 여유가 적다)
#   1순위 /app/data — 쓰기 가능하면 tar 조각 옆에 제자리 해제라 사본이 0
#   2순위 /app/scratch — 사업단 파드의 작업용 볼륨 (2026-09-28 실측: 쓰기 가능, 여유 950 GB). 파드가 여기에 저장소를 clone 하고 HF 캐시도 여기로 지정한다
#   3순위 컨테이너 임시 디스크
WORK_ROOT=${WORK_ROOT:-$(writable /app/data && echo /app/data || { writable /app/scratch && echo /app/scratch/h200_work || echo "${TMPDIR:-/tmp}/h200_work"; })}
case $WORK_ROOT in /app/output*) echo "!!! WORK_ROOT 가 /app/output 을 가리킴 — 결과 전용 경로다. 중단"; exit 1;; esac
mkdir -p "$WORK_ROOT"; XROOT=$WORK_ROOT/h200_extracted   # 제자리 해제가 불가능할 때만 쓰는 대체 경로
# 작업 사이에 남을 수 있는 경로면 해제본을 지우지 않는다. /app/data 와 /app/scratch 는 결과 볼륨이 아니고 넉넉하므로 남겨서 다음 작업이 재사용하게 한다
case $WORK_ROOT in /app/data*|/app/scratch*) WORK_PERSISTENT=1;; *) WORK_PERSISTENT=0;; esac
# 단계마다 실제로 읽는 것만 푼다 (임시 디스크로 풀 때만 적용. /app/data 제자리 해제는 한 번에 전부 풀어 두고 모든 단계가 재사용한다)
#   label 교사 추론 = 풀 이미지 + 교사 가중치 | train 학생 학습 = 풀 이미지만 (교사 가중치 불필요) | eval 평가 = 평가셋만 (풀 이미지·교사 가중치 불필요)
case $MODE in label) NEED_MEMBERS="pool models";; train) NEED_MEMBERS="pool";; eval) NEED_MEMBERS="eval";; grid) NEED_MEMBERS="pool eval";; *) NEED_MEMBERS="";; esac
# 토큰: 기본은 이슈 명령 인자(hf_...). 대안으로 /app/data/hf_token.txt 파일도 읽는다
for tf in /app/data/hf_token.txt "$DATA_ROOT/hf_token.txt"; do [ -z "${HF_TOKEN:-}" ] && [ -f "$tf" ] && export HF_TOKEN=$(tr -d '[:space:]' < "$tf") && echo "[setup] HF token loaded from $tf"; done
# HF 가중치 캐시(학생 7.5 GB)는 WORK_ROOT 에 둔다. /app/data 가 쓰기 가능하면 다음 작업이 재다운로드하지 않고, 아니면 파드와 함께 사라진다. /app/output 에는 두지 않는다
export HF_HOME=${HF_HOME:-$WORK_ROOT/hf}
case $HF_HOME in /app/output*) echo "[setup] HF_HOME 이 /app/output 을 가리켜 $WORK_ROOT/hf 로 되돌림 (결과 전용 경로)"; export HF_HOME=$WORK_ROOT/hf;; esac
if [ "$MODE" = check ]; then   # 압축 해제 위치·쓰기 권한·여유 용량만 확인한다. GPU·모델·데이터 불필요
  echo "=== 경로 점검 ($(date '+%F %T')) ==="
  for d in /app/data /app/output "$WORK_ROOT"; do
    if [ -d "$d" ]; then echo "  $d  →  $(writable "$d" && echo '쓰기 가능' || echo '읽기 전용'),  여유 $(freegb "$d") GB,  사용 $(usedgb "$d") GB"
    else echo "  $d  →  없음"; fi; done
  echo "=== 해제 계획 ==="
  PD=""; for d in /app/data "$DATA_SRC"; do [ -d "$d" ] && [ -z "$PD" ] && PD=$(find "$d" -maxdepth 3 -name "depthlm_distill_h200_app_data.tar.part_00" -printf "%h\n" 2>/dev/null | head -1 || true); done
  PX=""; for d in "$XROOT" /app/data "$DATA_SRC"; do [ -d "$d" ] && [ -z "$PX" ] && PX=$(find "$d" -maxdepth 4 -type d -name "depthlm_distill_h200" 2>/dev/null | head -1 || true); done
  if [ -n "$PX" ] && [ -d "$PX/pool" ]; then echo "  이미 풀려 있음: $PX  ($(usedgb "$PX") GB, 풀 이미지 $(find "$PX/pool" -type f | wc -l) 장) → 추가 해제 없음. 지난 작업의 해제본이 남아 있다는 뜻이다"
  elif [ -n "$PD" ]; then echo "  조각 위치: $PD  ($(ls "$PD"/depthlm_distill_h200_app_data.tar.part_* 2>/dev/null | wc -l)/16 개, 체크섬 $([ -f "$PD/SHA256SUMS_parts" ] && echo 있음 || echo 없음))"
    if writable "$PD"; then echo "  → 제자리 해제 (사본 없음). $PD 에 30 GB 추가, 여유 $(freegb "$PD") GB"
    else echo "  → 조각 폴더가 읽기 전용이므로 $XROOT 에 해제 (30 GB). 여유 $(freegb "$WORK_ROOT") GB"
         echo "  판정: 결과 볼륨(/app/output)이 아니므로 100 GB 할당량과 무관하다"
         echo "  참고: 이 볼륨이 작업 사이에 유지되지 않으면 작업마다 다시 푼다 (label 29.5 GB, train 5.5 GB, eval 0.4 GB. 30 GB 기준 약 2 분)"; fi
  else echo "  조각(depthlm_distill_h200_app_data.tar.part_00)을 /app/data 아래에서 찾지 못함"; fi
  echo "DATA_ROOT=$DATA_ROOT"; echo "WORK_ROOT=$WORK_ROOT"; echo "XROOT=$XROOT"; echo "OUT_ROOT=$OUT_ROOT (결과 전용)"; echo "HF_HOME=$HF_HOME"
  echo "=== 교사 라벨 (학생 학습이 읽을 것) ==="
  for pl in mixed indoor outdoor; do
    LF=""; for c in "$OUT_ROOT/labels/$pl/teacher_labels.parquet" "$WORK_ROOT/labels/$pl/teacher_labels.parquet" "pools/$pl/teacher_labels.parquet"; do
      { [ -z "$LF" ] && [ -f "$c" ] && LF=$c; } || true; done
    if [ -n "$LF" ]; then echo "  $pl  →  $LF  ($(du -h "$LF" | cut -f1))"
    else echo "  $pl  →  없음.  bash run.sh label $pl 을 먼저 요청할 것"; fi; done
  echo "=== /app/output 점검 (결과만 있어야 한다) ==="
  for bad in data hf h200_extracted data_pack; do [ -e "$OUT_ROOT/$bad" ] && echo "  !!! $OUT_ROOT/$bad 가 있다 ($(usedgb "$OUT_ROOT/$bad") GB) — 결과가 아니므로 지울 것" || true; done
  ls -A "$OUT_ROOT" 2>/dev/null | head -20 | sed 's/^/  /'
  exit 0
fi
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
if [ -n "${HF_TOKEN:-}" ]; then TOKRC=0; python - "$DATA_REPO" > "$OUT_ROOT/token_check.txt" 2>&1 <<'PYT' || TOKRC=$?
import sys; from huggingface_hub import HfApi
api = HfApi(); who = "?"
try: who = api.whoami()["name"]
except Exception as e: print(f"!!! [setup] 토큰 무효(만료·오타·삭제됨): {type(e).__name__}: {str(e).strip().splitlines()[-1][:120]}"); sys.exit(3)
for kind, rid in (("model", "facebook/DepthLM"), ("dataset", sys.argv[1])):
    try: (api.model_info if kind == "model" else api.dataset_info)(rid); print(f"[setup] 토큰({who}) → {rid} 접근 OK")
    except Exception as e: print(f"!!! [setup] 토큰({who}) → {rid} 접근 실패: {type(e).__name__} (라이선스 동의·비공개 저장소 권한 확인)")
PYT
  cat "$OUT_ROOT/token_check.txt" | tee -a "$LOG"; rm -f "$OUT_ROOT/token_check.txt"
  if [ "$TOKRC" = "3" ]; then unset HF_TOKEN   # 무효한 토큰을 그대로 두면 공개 모델 다운로드까지 401 로 막힌다
    if [ "$MODE" = smoke ]; then say "!!! [setup] 토큰 없이 스모크 계속 (학생 모델은 공개). 새 토큰(만료 없음)으로 다시 요청할 것"
    else say "!!! [setup] 토큰이 무효라 데이터·교사 다운로드가 불가능 → 종료. 새 토큰(만료 없음)으로 다시 요청할 것"; exit 1; fi; fi
else say "[setup] HF 토큰 없음 — 학생(공개)은 다운로드 가능. 교사는 로컬 가중치(/app/data 조각)가 있으면 토큰 불필요"; fi
# 스모크는 저장소에 포함된 합성 40장으로만 돌기 때문에 30 GB 팩이 전혀 필요 없다. DATA_ROOT 를 미리 가리켜 아래 데이터 확보 블록을 모두 건너뛴다
if [ "$MODE" = smoke ]; then export DATA_ROOT=$PWD/smoke/data; echo "[setup] 스모크: 저장소의 합성 데이터만 사용 ($DATA_ROOT) — tar 를 풀지 않는다"; fi
# 관리자 부담 최소화: 드라이브의 조각(depthlm_distill_h200_app_data.tar.part_*)을 /app/data 아래 아무 폴더에 받아 두기만 하면 스크립트가 검증하고 한 번 푼다 (제자리, 불가능하면 $XROOT — /app/output 은 아니다)
if [ ! -d "$DATA_ROOT/pool" ]; then
  PRE=""; for d in "$XROOT" /app/data "$DATA_SRC"; do [ -d "$d" ] && [ -z "$PRE" ] && PRE=$(find "$d" -maxdepth 4 -type d -name "depthlm_distill_h200" 2>/dev/null | head -1 || true); done
  if [ -n "$PRE" ] && have_members "$PRE" "${NEED_MEMBERS:-pool eval}"; then export DATA_ROOT=$PRE; echo "[data] 이미 풀린 폴더 사용: $DATA_ROOT ($(ls "$PRE" | tr '\n' ' '))"
  else
    { [ -n "$PRE" ] && echo "[data] 부분 해제본 발견: $PRE ($(ls "$PRE" | tr '\n' ' ')) — 이 단계가 필요한 '${NEED_MEMBERS:-pool eval}' 중 빠진 것만 채운다"; } || true
    PDIR=""; for d in /app/data "$DATA_SRC"; do [ -d "$d" ] && [ -z "$PDIR" ] && PDIR=$(find "$d" -maxdepth 3 -name "depthlm_distill_h200_app_data.tar.part_00" -printf "%h\n" 2>/dev/null | head -1 || true); done   # set -e/pipefail 안전
    # 관리자가 드라이브 폴더를 통째로 받으면 zip(여러 개일 수 있음, 조각이 나뉘어 들어감)으로 온다 → zip 안의 조각을 순서대로 tar 로 바로 흘려 넣어 풀고(중간 복사본 없음) SHA256 은 흘리면서 검증
    if [ -z "$PDIR" ]; then
      ZIPS=""; for d in /app/data "$DATA_SRC"; do [ -d "$d" ] && ZIPS="$ZIPS $(find "$d" -maxdepth 3 -name "*.zip" 2>/dev/null | tr '\n' ' ' || true)"; done
      ZIPS=$(python - $ZIPS <<'PYL'
import sys, zipfile
print(" ".join(z for z in sys.argv[1:] if zipfile.is_zipfile(z) and any(n.endswith("depthlm_distill_h200_app_data.tar.part_00") or n.endswith("depthlm_distill_h200_app_data.tar.part_15") for n in zipfile.ZipFile(z).namelist())))
PYL
)
      if [ -n "$ZIPS" ]; then
        ZD=$(dirname "$(echo $ZIPS | cut -d' ' -f1)"); if touch "$ZD/.write_test" 2>/dev/null; then rm -f "$ZD/.write_test"; XDIR=$ZD; echo "[data] zip 발견: $ZIPS → 폴더가 쓰기 가능, 제자리에서 풀기"; else XDIR=$XROOT; echo "[data] zip 발견: $ZIPS → 읽기 전용, $XROOT 에 풀기 (30 GB). /app/output 에는 풀지 않는다"; fi
        mkdir -p "$XDIR"; needgb "$XDIR" "$NEED_GB"; python - "$XDIR" $ZIPS <<'PYZ' && export DATA_ROOT=$XDIR/depthlm_distill_h200 && { [ "$XDIR" = "$XROOT" ] && EXTRACTED_HERE=1 || true; } && echo "[data] 풀기 완료: 풀 이미지 $(find "$DATA_ROOT/pool" -type f | wc -l), 평가 파일 $(find "$DATA_ROOT/eval" -type f | wc -l), 교사 가중치 조각 $(ls "$DATA_ROOT/models/DepthLM" | grep -c safetensors)" || { echo "!!! [data] zip 에서 풀기 실패 (조각 누락 또는 SHA256 불일치)"; exit 1; }
import sys, zipfile, hashlib, subprocess, os, re, shutil
xdir, zips = sys.argv[1], sys.argv[2:]; members = {}; sums = {}
for z in zips:
    zf = zipfile.ZipFile(z)
    for n in zf.namelist():
        b = os.path.basename(n); m = re.match(r"depthlm_distill_h200_app_data\.tar\.part_(\d+)$", b)
        if m: members[int(m.group(1))] = (zf, n)
        if b == "SHA256SUMS_parts": sums = {l.split()[1].lstrip("*"): l.split()[0] for l in zf.read(n).decode().splitlines() if l.strip()}
idx = sorted(members); print(f"[data] zip 안 조각 {len(idx)} 개 (part_{idx[0]:02d}..part_{idx[-1]:02d}), 체크섬 {len(sums)} 개", flush=True)
if idx != list(range(16)) or len(sums) < 16: print("!!! [data] zip 안 조각이 16 개가 아니거나 체크섬 파일이 없음 — 업로드가 덜 됐을 수 있음. 다 올라간 뒤 다시 요청할 것"); sys.exit(1)
tmpx = os.path.join(xdir, f".extracting_{os.getpid()}"); shutil.rmtree(tmpx, ignore_errors=True); os.makedirs(tmpx)
tar = subprocess.Popen(["tar", "-xf", "-", "-C", tmpx], stdin=subprocess.PIPE); bad = []
for i in idx:
    zf, n = members[i]; h = hashlib.sha256()
    with zf.open(n) as f:
        while True:
            chunk = f.read(16 << 20)
            if not chunk: break
            h.update(chunk); tar.stdin.write(chunk)
    name = f"depthlm_distill_h200_app_data.tar.part_{i:02d}"
    if sums and sums.get(name) != h.hexdigest(): bad.append(name)
tar.stdin.close(); rc = tar.wait()
if bad or rc != 0:
    print(f"!!! [data] SHA256 불일치 {bad} / tar 종료코드 {rc} — 임시 폴더 정리함"); shutil.rmtree(tmpx, ignore_errors=True); sys.exit(1)
os.rename(os.path.join(tmpx, "depthlm_distill_h200"), os.path.join(xdir, "depthlm_distill_h200")); shutil.rmtree(tmpx, ignore_errors=True)
print("[data] zip 조각 SHA256 검증 통과 (흘리면서 검증)")
PYZ
      fi
    elif [ -d "$PDIR/depthlm_distill_h200/pool" ]; then export DATA_ROOT=$PDIR/depthlm_distill_h200; echo "[data] 이전 작업이 제자리에 풀어 둔 것을 사용: $DATA_ROOT"
    elif [ -n "$PDIR" ]; then
      NP=$(ls "$PDIR"/depthlm_distill_h200_app_data.tar.part_* | wc -l)
      [ "$NP" = 16 ] && [ -f "$PDIR/SHA256SUMS_parts" ] || { echo "!!! [data] 조각 $NP/16 개, 체크섬 파일 $([ -f "$PDIR/SHA256SUMS_parts" ] && echo 있음 || echo 없음) — 아직 업로드 중일 수 있음. 다 올라간 뒤 다시 요청할 것"; exit 1; }
      if writable "$PDIR"; then XDIR=$PDIR; echo "[data] 조각 발견: $PDIR (16 개). 폴더가 쓰기 가능 → 제자리에서 풀기 (사본 없음)"
      else XDIR=$XROOT; echo "[data] 조각 발견: $PDIR (16 개). 폴더가 읽기 전용 → $XROOT 에 풀기 (30 GB). /app/output 에는 풀지 않는다"
           { [ "$WORK_PERSISTENT" = 1 ] && echo "[data] 결과 볼륨이 아닌 작업 볼륨이므로 /app/output 용량과 무관하다" || echo "[data] 컨테이너 임시 디스크라 파드가 끝나면 사라진다 — 다음 작업이 다시 푼다"; }; fi
      if [ -n "$PRE" ] && writable "$(dirname "$PRE")"; then XDIR=$(dirname "$PRE"); echo "[data] 기존 부분 해제본을 채운다: $XDIR/depthlm_distill_h200"; fi
      mkdir -p "$XDIR"; NG=$NEED_GB   # 임시 디스크에 단계별로 부분 해제할 때는 요구 공간도 줄어든다
      if [ -z "${NEED_GB_EXPLICIT:-}" ] && [ "$XDIR" = "$XROOT" ]; then case $MODE in train) NG=8;; eval) NG=2;; esac; fi
      needgb "$XDIR" "$NG"
      (cd "$PDIR" && sha256sum -c --quiet SHA256SUMS_parts) && echo "[data] 조각 SHA256 검증 통과" || { echo "!!! [data] 조각 SHA256 불일치 — 업로드가 덜 됐거나 깨짐. 다시 받을 것"; exit 1; }
      TMPX=$XDIR/.extracting_$$; rm -rf "$TMPX"; mkdir -p "$TMPX"   # 임시 폴더에 풀고 성공했을 때만 최종 이름으로 (중간에 죽어도 반쪽짜리 폴더가 남지 않음)
      MEM=""   # 임시 디스크로 푸는 경우에만 이 단계가 읽는 폴더로 한정 (train 5.5 GB, eval 0.4 GB, label 29.5 GB, 전체 30 GB)
      if [ "$XDIR" = "$XROOT" ] && [ -n "$NEED_MEMBERS" ]; then   # data/smoke/all 은 NEED_MEMBERS 가 비어 있어 전체를 푼다
        for m in $NEED_MEMBERS; do MEM="$MEM depthlm_distill_h200/$m"; done
        echo "[data] $MODE 단계가 읽는 것만 풀기:$MEM"; fi
      merge_in() {   # 대상이 없으면 그대로 옮기고, 부분 해제본이 있으면 빠진 폴더만 채운다
        if [ -d "$XDIR/depthlm_distill_h200" ]; then
          for e in "$TMPX/depthlm_distill_h200"/* "$TMPX/depthlm_distill_h200"/.[!.]*; do
            [ -e "$e" ] || continue; mv -n "$e" "$XDIR/depthlm_distill_h200/" 2>/dev/null || true; done
          rm -rf "$TMPX"
        else mv "$TMPX/depthlm_distill_h200" "$XDIR/depthlm_distill_h200" || return 1; fi
        [ -d "$XDIR/depthlm_distill_h200" ]; }
      if cat "$PDIR"/depthlm_distill_h200_app_data.tar.part_* | tar -xf - -C "$TMPX" $MEM && merge_in; then
        rm -rf "$TMPX"; export DATA_ROOT=$XDIR/depthlm_distill_h200
        { [ "$XDIR" = "$XROOT" ] && EXTRACTED_HERE=1 || true; }
        # 관리자 팩에는 풀 v4 새 이미지가 이미 들어 있다(extra_done.txt). 부분 해제에는 그 표시 파일이 없으므로 대신 남겨 추가 팩을 다시 덧씌우지 않게 한다
        { [ -n "$MEM" ] && touch "$DATA_ROOT/.extra_done" || true; }
        echo "[data] 풀기 완료: $DATA_ROOT ($(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1)) — 풀 이미지 $(find "$DATA_ROOT/pool" -type f 2>/dev/null | wc -l) 장, 평가 파일 $(find "$DATA_ROOT/eval" -type f 2>/dev/null | wc -l) 개, 교사 가중치 조각 $(ls "$DATA_ROOT/models/DepthLM" 2>/dev/null | grep -c safetensors || true) 개"
      else rm -rf "$TMPX"; echo "!!! [data] 풀기 실패 — 임시 폴더 정리함. 다시 요청할 것"; exit 1; fi
    fi
  fi
fi
# 데이터 확보 순서: ① /app/data 에 풀려 있음 → ② 이전 작업이 $XROOT 에 풀어 둠 → ③ /app/data 의 tar 분할본 → ④ HF 비공개 데이터셋(DATA_REPO)에서 토큰으로 내려받음
# 추가 팩(depthlm_distill_data_extra.tar = 실내 풀 v4 의 새 이미지 205 장)은 본 팩 위에 한 번만 덧씌운다 (.extra_done 표시)
hf_fetch() { local out=$1; shift; python - "$DATA_REPO" "$out" "$@" <<'PYD'
import sys, time; from huggingface_hub import snapshot_download
repo, out, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
for a in range(3):
    try: snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=pats); print(f"[data] 다운로드 완료: {pats}"); break
    except Exception as e:
        print(f"!!! [data] 시도 {a+1} 실패: {type(e).__name__}: {str(e)[:120]}")
        if type(e).__name__ in ("RepositoryNotFoundError", "GatedRepoError"): sys.exit(1)   # 권한·이름 문제는 재시도 무의미
        time.sleep(30)
else: sys.exit(1)
PYD
}
if [ ! -d "$DATA_ROOT/pool" ]; then
  if [ -d "$XROOT/pool" ]; then export DATA_ROOT=$XROOT
  else
    PACK=""; for d in "$DATA_ROOT" /app/data "$DATA_SRC"; do [ -d "$d" ] && [ -z "$PACK" ] && PACK=$(find "$d" -maxdepth 3 -name "depthlm_distill_data.tar.part_aa" -printf "%h\n" 2>/dev/null | head -1 || true); done
    if [ -z "$PACK" ] && [ -n "${HF_TOKEN:-}" ]; then
      echo "[data] $DATA_REPO 에서 데이터 팩 다운로드 (6 GB)"; mkdir -p "$WORK_ROOT/data_pack"
      hf_fetch "$WORK_ROOT/data_pack" "depthlm_distill_data.tar.part_*" "SHA256SUMS" && PACK=$WORK_ROOT/data_pack || echo "!!! [data] 다운로드 실패 — 토큰이 $DATA_REPO 를 읽을 수 있는지 확인"
    fi
    if [ -n "$PACK" ]; then
      [ -f "$PACK/SHA256SUMS" ] && { (cd "$PACK" && sha256sum -c --quiet SHA256SUMS) && echo "[data] SHA256 검증 통과" || { echo "!!! [data] SHA256 불일치 — 분할본이 깨짐"; exit 1; }; }
      mkdir -p "$XROOT"; needgb "$XROOT" 8; cat "$PACK"/depthlm_distill_data.tar.part_* | tar -xf - -C "$XROOT" --strip-components=1 && { export DATA_ROOT=$XROOT; EXTRACTED_HERE=1; }
      { [ "$PACK" = "$WORK_ROOT/data_pack" ] && rm -rf "$WORK_ROOT/data_pack" || true; }; echo "[data] 풀기 완료: $(find "$DATA_ROOT/pool" -type f | wc -l) 풀 이미지, $(find "$DATA_ROOT/eval" -type f | wc -l) 평가 파일"
    fi
  fi
fi
if [ "$MODE" != smoke ] && [ -d "$DATA_ROOT/pool" ] && [ ! -f "$DATA_ROOT/.extra_done" ] && [ ! -f "$DATA_ROOT/extra_done.txt" ]; then   # 추가 팩 (실내 v4 새 이미지). 관리자 묶음에는 이미 포함(extra_done.txt)
  EX=""; for d in /app/data "$DATA_SRC" "$DATA_ROOT"; do [ -d "$d" ] && [ -z "$EX" ] && EX=$(find "$d" -maxdepth 3 -name "depthlm_distill_data_extra.tar" -printf "%h\n" 2>/dev/null | head -1 || true); done
  if [ -z "$EX" ] && [ -n "${HF_TOKEN:-}" ]; then mkdir -p "$WORK_ROOT/data_pack"; hf_fetch "$WORK_ROOT/data_pack" "depthlm_distill_data_extra.tar" "SHA256SUMS_extra" >/dev/null 2>&1 || true; [ -f "$WORK_ROOT/data_pack/depthlm_distill_data_extra.tar" ] && EX=$WORK_ROOT/data_pack; fi
  if [ -n "$EX" ] && [ -w "$DATA_ROOT" ]; then
    [ -f "$EX/SHA256SUMS_extra" ] && { (cd "$EX" && sha256sum -c --quiet SHA256SUMS_extra) || { echo "!!! [data] 추가 팩 SHA256 불일치"; exit 1; }; }
    tar -xf "$EX/depthlm_distill_data_extra.tar" -C "$DATA_ROOT" --strip-components=1 && touch "$DATA_ROOT/.extra_done" && echo "[data] 추가 팩 풀기 완료 ($(tar -tf "$EX/depthlm_distill_data_extra.tar" | grep -cE '\.(png|jpg)$') 장)"
    { [ "$EX" = "$WORK_ROOT/data_pack" ] && rm -rf "$WORK_ROOT/data_pack" || true; }
  else echo "!!! [data] 추가 팩(depthlm_distill_data_extra.tar) 없음 — 실내 풀 v4 의 새 이미지 205 장이 없어 실내 라벨링·격자는 실패함 (HF 데이터셋에 올렸는지 확인)"; fi
fi
# 모델 가중치가 /app/data/models 에 있으면 그것을 쓰고, 없으면 Hugging Face 에서 내려받음 (인터넷 필요)
[ -d "$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct" ] && export STUDENT_MODEL=$DATA_ROOT/models/Qwen2.5-VL-3B-Instruct
[ -d "$DATA_ROOT/models/DepthLM" ] && export TEACHER_MODEL=$DATA_ROOT/models/DepthLM
for d in /app/data "$DATA_SRC"; do [ -d "$d" ] || continue   # 관리자가 가중치를 tar 없이 그대로 받아 둔 경우: /app/data 아래 어느 폴더든 models/DepthLM 을 찾는다
  [ -z "${TEACHER_MODEL:-}" ] && t=$(find -L "$d" -maxdepth 4 -type f -name "model.safetensors.index.json" -path "*DepthLM*" -printf "%h\n" 2>/dev/null | head -1 || true) && [ -n "$t" ] && export TEACHER_MODEL=$t
  [ -z "${STUDENT_MODEL:-}" ] && q=$(find -L "$d" -maxdepth 4 -type d -name "Qwen2.5-VL-3B-Instruct" 2>/dev/null | head -1 || true) && [ -n "$q" ] && export STUDENT_MODEL=$q; done
echo "[setup] 교사 가중치: ${TEACHER_MODEL:-facebook/DepthLM (HF, 토큰 필요)} | 학생 가중치: ${STUDENT_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct (HF 공개)}"
say "MODE=$MODE POOL=$POOL COND=$COND FOCAL=$FOCAL DATA_ROOT=$DATA_ROOT OUT_ROOT=$OUT_ROOT"
if [ "$MODE" = data ]; then   # 압축 해제만 하고 끝낸다. 한 번 해 두면 label/train/eval 작업이 그대로 재사용한다
  [ -d "$DATA_ROOT/pool" ] && [ -d "$DATA_ROOT/eval" ] || { say "!!! [data] 준비 실패 — DATA_ROOT($DATA_ROOT) 에 pool/ eval/ 이 없다"; exit 1; }
  say "[data] 준비 완료: 풀 이미지 $(find "$DATA_ROOT/pool" -type f | wc -l) 장, 평가 파일 $(find "$DATA_ROOT/eval" -type f | wc -l) 개"
  say "[data] 위치: $DATA_ROOT  (여유 $(freegb "$DATA_ROOT") GB)"
  if [ "$WORK_PERSISTENT" = 0 ] && [ "$EXTRACTED_HERE" = 1 ]; then
    say "!!! [data] /app/data 가 읽기 전용이라 컨테이너 임시 디스크($XROOT)에 풀렸다. 파드가 끝나면 사라지므로 이 작업은 아무것도 남기지 못한다"
    say "!!! [data] 담당자에게 /app/data 쓰기 권한이나 작업용 볼륨(예: /app/scratch)을 요청할 것. 없으면 data 모드를 건너뛰고 label/train/eval 을 바로 요청하면 된다 (각 작업이 시작할 때 임시 디스크에 필요한 만큼만 풀고 /app/output 은 건드리지 않는다)"
    exit 1
  fi
  case $DATA_ROOT in
    "$XROOT"/*)   # 조각 폴더가 읽기 전용이라 작업 볼륨에 풀었다
      say "[data] $DATA_ROOT 에 남았다 — 작업 볼륨($WORK_ROOT)이고 결과 볼륨(/app/output)과 별개다"
      say "[data] tar 조각은 /app/data 에 그대로 둘 것. 작업 볼륨이 작업 사이에 비워지면 다시 풀어야 한다"
      say "[data] 이 작업을 반복할 필요는 없다. 다음 작업으로 bash run.sh check 를 내서 '이미 풀려 있음' 이 나오는지만 보면 된다"
      say "[data] 나오지 않으면 이 볼륨은 작업마다 비워지는 것이고, label/train/eval 이 각자 필요한 만큼만 스스로 푼다 (data 는 더 낼 필요 없음)";;
    *)            # tar 조각 옆 제자리
      say "[data] $DATA_ROOT 에 tar 조각 옆 제자리로 남았다 — 사본이 없다"
      say "[data] tar 조각(약 29 GB)은 이제 지워도 된다: 관리자에게 depthlm_distill_h200_app_data.tar.part_* 삭제 요청";;
  esac
  say "[data] 다음 작업으로 요청할 것 →  bash run.sh label $POOL"; exit 0
fi
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG" || say "nvidia-smi 없음"
GPU_MB=$(python -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/2**20) if torch.cuda.is_available() else 0)")   # nvidia-smi 는 컨테이너에서 메모리 값을 못 줄 수 있어 torch 로 판정
DEVS=($(nvidia-smi -L 2>/dev/null | grep -oE "MIG-[0-9a-f-]+" || true)); NDEV=${#DEVS[@]}   # MIG 슬라이스가 여러 개 보이면 셀을 슬라이스별로 분배
NPROC_TRAIN=${NPROC:-$([ "${GPU_MB:-0}" -gt 100000 ] && echo 8 || { [ "$NDEV" -gt 1 ] && echo "$NDEV" || echo 1; })}; NPROC_LABEL=${NPROC_LABEL:-$([ "${GPU_MB:-0}" -gt 100000 ] && echo 4 || echo 1)}
say "GPU ${GPU_MB} MiB, MIG 장치 $NDEV → 학습 병렬 $NPROC_TRAIN, 라벨링 병렬 $NPROC_LABEL"
if [ "$MODE" = label ] || [ "$MODE" = all ]; then [ "${GPU_MB:-0}" -ge 28000 ] || say "!!! 장치 메모리 ${GPU_MB} MiB < 28 GB — 교사(12B)가 들어가지 않아 라벨링은 실패함. GPU 할당량 7(통째)로 요청할 것"; fi
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a "$LOG"
if [ "$MODE" = smoke ]; then
  say "[smoke] 학습 30 스텝 (Qwen2.5-VL-3B 다운로드 포함)"
  python -u experiments/20_train_student.py --cond soft --steps 30 --accum 8 --focal "$FOCAL" --labels pools/smoke/teacher_labels.parquet --pools configs/pool_smoke.yaml --rows pools/smoke/rows_smoke.parquet --tag _smoke 2>&1 | grep -vE "^\[transformers\]" | tee -a "$LOG"
  say "[smoke] 평가 3 px (데이터셋당 1 이미지 1 픽셀)"
  python -u experiments/21_eval_student.py --tag smoke --adapter "$OUT_ROOT/checkpoints/soft_smoke" --focal "$FOCAL" --eval_set small --limit_img 1 --per_max 1 2>&1 | grep -vE "^\[transformers\]" | tee -a "$LOG"
  say "[smoke] 완료. 결과: $OUT_ROOT/eval/eval_smoke.parquet, 어댑터: $OUT_ROOT/checkpoints/soft_smoke"; exit 0
fi
if [ "$MODE" = all ]; then   # 한 이슈로 전체 체인. 각 단계는 하위 실행이라 하나가 실패해도 다음으로 넘어간다. 라벨링은 빠진 쌍만 한다
  say "!!! [all] 권장하지 않음 — 2026-09 실행에서 35 시간 뒤 강제 종료됐고 어디까지 됐는지 로그로 추적하기 어려웠다. label → train → eval 을 따로 요청할 것"
  say "[all] 라벨링 mixed (풀 v4 교체분 1,160 px)"; H200_CHILD=1 bash run.sh label mixed || say "!!! [all] 라벨링 실패 mixed"
  for c in soft hard; do say "[all] 격자 mixed $c"; H200_CHILD=1 bash run.sh grid mixed $c || say "!!! [all] 격자 실패 mixed $c"; done
  for p in indoor outdoor; do [ -n "${HF_TOKEN:-}" ] || [ -n "${TEACHER_MODEL:-}" ] || say "!!! [all] 토큰도 로컬 교사 가중치도 없음 — $p 라벨링은 실패할 것"; say "[all] 라벨링 $p"; H200_CHILD=1 bash run.sh label $p || say "!!! [all] 라벨링 실패 $p"; done
  # 라벨링이 둘 다 끝났으면 토큰 사본을 지운다 (이후 격자는 토큰 불필요). /app/data 가 읽기 전용이면 관리자에게 삭제 요청. 실제 무효화는 HF 계정에서 Revoke 해야 한다
  if [ -f "$OUT_ROOT/labels/indoor/teacher_labels.parquet" ] && [ -f "$OUT_ROOT/labels/outdoor/teacher_labels.parquet" ]; then
    for tf in /app/data/hf_token.txt "$DATA_ROOT/hf_token.txt"; do [ -f "$tf" ] && { rm -f "$tf" 2>/dev/null && say "[all] 토큰 파일 삭제됨: $tf" || say "!!! [all] 토큰 파일을 지우지 못함(읽기 전용): $tf — 관리자에게 삭제 요청"; }; done
    unset HF_TOKEN; say "[all] 라벨링 완료 — 이후 단계는 토큰을 쓰지 않음. HF 설정에서 토큰을 Revoke 할 것"
  else say "!!! [all] 라벨이 둘 다 없어 토큰 파일을 남겨 둠(재실행용)"; fi
  for p in indoor outdoor; do for c in soft hard; do say "[all] 격자 $p $c"; H200_CHILD=1 bash run.sh grid $p $c || say "!!! [all] 격자 실패 $p $c"; done; done
  say "[all] 완료. 결과 zip: $(ls "$OUT_ROOT"/results_*.zip 2>/dev/null | tr '\n' ' ')"; exit 0
fi
if [ "$MODE" = label ]; then   # 교사 라벨링: 저장소 라벨 + 이전 part 를 base 로 두고 todo 중 빠진 쌍만 라벨링 → teacher_labels.parquet (완전본). VRAM 28-30 GB
  [ -d "$DATA_ROOT/pool" ] || { say "!!! DATA_ROOT 에 pool/ 없음"; exit 1; }; LD=$OUT_ROOT/labels/$POOL; mkdir -p "$LD"
  { [ -f "$WORK_ROOT/labels/$POOL/teacher_labels.parquet" ] && cp -n "$WORK_ROOT/labels/$POOL/teacher_labels.parquet" "$LD/part_persisted.parquet" 2>/dev/null && say "[label] 이전 작업의 영속 사본을 base 로 이어서 라벨링"; } || true
  python - "$POOL" "$NPROC_LABEL" "$LD" <<'PYS' | tee -a "$LOG"
import sys, os, glob, pandas as pd; pool, n, ld = sys.argv[1], int(sys.argv[2]), sys.argv[3]
todo = pd.read_parquet(f"pools/{pool}/todo_label.parquet")[["image_id", "pixel_index"]].drop_duplicates()
srcs = [p for p in [f"pools/{pool}/teacher_labels.parquet"] + sorted(glob.glob(f"{ld}/part_*.parquet")) if os.path.exists(p)]
have = pd.concat([pd.read_parquet(p) for p in srcs], ignore_index=True).drop_duplicates(["image_id", "pixel_index"]) if srcs else pd.DataFrame(columns=list(todo.columns))
have = have.merge(todo, on=["image_id", "pixel_index"]); have.to_parquet(f"{ld}/base.parquet", index=False)
m = todo.merge(have[["image_id", "pixel_index"]], on=["image_id", "pixel_index"], how="left", indicator=True); miss = m[m._merge == "left_only"][["image_id", "pixel_index"]]
for f in glob.glob(f"{ld}/todo_*.parquet"): os.remove(f)
k = min(n, max(1, (len(miss) + 63) // 64)) if len(miss) else 0   # 조각당 최소 64 px
for i in range(k): miss.iloc[i::k].to_parquet(f"{ld}/todo_{i}.parquet", index=False)
open(f"{ld}/n_shards.txt", "w").write(str(k)); print(f"[label] {pool}: 필요 {len(todo)} px, 보유 {len(have)} px, 라벨링 {len(miss)} px → 조각 {k}")
PYS
  K=$(cat "$LD/n_shards.txt")
  if [ "$K" -gt 0 ]; then
    for i in $(seq 0 $((K-1))); do
      ( for a in 1 2 3; do python -u experiments/11_label_teacher.py --pool pools/$POOL/pool.jsonl --image_folder "$DATA_ROOT" --todo "$LD/todo_$i.parquet" --out "$LD/part_$i.parquet" --chunk 8 > "$LD/label_$i.log" 2>&1 && break; sleep 30; done ) &
    done; wait
  fi
  MRC=0; python - "$POOL" "$LD" > "$LD/merge.txt" 2>&1 <<'PYS' || MRC=$?
import sys, glob, pandas as pd; pool, ld = sys.argv[1], sys.argv[2]
todo = pd.read_parquet(f"pools/{pool}/todo_label.parquet")[["image_id", "pixel_index"]].drop_duplicates()
d = pd.concat([pd.read_parquet(p) for p in [f"{ld}/base.parquet"] + sorted(glob.glob(f"{ld}/part_*.parquet"))], ignore_index=True).drop_duplicates(["image_id", "pixel_index"])
d = d.merge(todo, on=["image_id", "pixel_index"]); d.to_parquet(f"{ld}/teacher_labels.parquet", index=False); miss = len(todo) - len(d)
print(f"[label] {pool} 병합 {len(d)} px / 필요 {len(todo)} px (부족 {miss}), 파싱 실패 {d.teacher_greedy1.isna().mean()*100:.2f}%, 질량 중앙 {d.teacher_mass.median():.3f}"); sys.exit(1 if miss else 0)
PYS
  cat "$LD/merge.txt" | tee -a "$LOG"; [ "$MRC" = 0 ] || { say "!!! [label] $POOL 라벨 부족 — 같은 명령을 다시 내면 이어서 라벨링"; exit 1; }
  say "[label] 완료: $LD/teacher_labels.parquet ($(du -h "$LD/teacher_labels.parquet" | cut -f1))"
  # /app/data 아래만 작업 사이에 남는다. /app/scratch 는 파드마다 비워지므로 사본을 둬도 다음 작업이 못 본다 (2026-09-28 실측)
  if case $WORK_ROOT in /app/data*) true;; *) false;; esac; then
    mkdir -p "$WORK_ROOT/labels/$POOL" && cp -f "$LD/teacher_labels.parquet" "$WORK_ROOT/labels/$POOL/teacher_labels.parquet" \
      && say "[label] 영속 사본: $WORK_ROOT/labels/$POOL/teacher_labels.parquet — 다음 파드가 /app/output 을 못 보더라도 학습이 여기서 읽는다" \
      || say "!!! [label] 영속 사본 복사 실패 (학습은 /app/output 경로로 계속 시도한다)"
  else say "!!! [label] /app/data 가 읽기 전용이라 영속 사본을 둘 수 없다 — 아래 커밋 절차를 반드시 할 것"; fi
  say "[label] /app/output 이 파드 사이에 넘어오지 않으면 이 parquet 를 내려받아 저장소의 pools/$POOL/teacher_labels.parquet 로 커밋할 것 (그러면 모든 파드가 저장소에서 읽는다)"
  python - "$LD/teacher_labels.parquet" "$OUT_ROOT/labels_${POOL}.zip" <<'PYZ' 2>&1 | tee -a "$LOG" || say "!!! [label] 라벨 zip 생성 실패 (parquet 는 그대로 있음)"
import sys, os, zipfile   # 결과 zip 은 labels/ 를 제외하므로 라벨만 따로 묶어 한 파일로 내려받게 한다
src, dst = sys.argv[1:3]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z: z.write(src, os.path.basename(src))
print(f"[label] 내려받을 파일: {dst}  {os.path.getsize(dst)/1e6:.1f} MB")
PYZ
  say "[label] 다음 작업으로 요청할 것 →  bash run.sh train $POOL soft"; exit 0
fi
# --- train / eval / grid ---  태그 = <cond>_<cell>_<pool>_f<focal>  (풀이 달라도 체크포인트·평가 파일이 겹치지 않음)
# train = 학습 8셀만, eval = 평가+표+zip 만, grid = 둘 다(호환용). 학습과 평가를 따로 요청하면 한 작업이 12 시간대로 끝나고 어디까지 됐는지 분명해진다
SUFFIX=_${POOL}_f${FOCAL}; ARMS=pools/$POOL/arms.json; PCFG=configs/pool_${POOL}.yaml
# 풀마다 판정에 쓰는 평가 세트만 평가한다 (실내 → iBims-1·NYUv2, 주행 → DDAD·nuScenes, 혼합 → 넷 다). EVAL_DATASETS 로 바꿀 수 있다
case $POOL in indoor) DEF_DS="ibims1 nyuv2";; outdoor) DEF_DS="ddad nuscenes";; *) DEF_DS="ibims1 nyuv2 ddad nuscenes";; esac; EVAL_DATASETS=${EVAL_DATASETS:-$DEF_DS}
# 교사 라벨 탐색 순서: ① 이 파드의 결과 ② /app/data 에 남긴 영속 사본 (파드 사이에 /app/output 이 안 넘어와도 살아남음) ③ 저장소에 커밋된 라벨
LABEL_TRIED="$OUT_ROOT/labels/$POOL/teacher_labels.parquet | $WORK_ROOT/labels/$POOL/teacher_labels.parquet | pools/$POOL/teacher_labels.parquet"
LABELS=""; for c in "$OUT_ROOT/labels/$POOL/teacher_labels.parquet" "$WORK_ROOT/labels/$POOL/teacher_labels.parquet" "pools/$POOL/teacher_labels.parquet"; do
  { [ -z "$LABELS" ] && [ -f "$c" ] && LABELS=$c; } || true; done
[ -f "$ARMS" ] || { say "!!! arms 파일 없음: $ARMS"; exit 1; }
if [ "$MODE" = eval ]; then [ -d "$DATA_ROOT/eval" ] || { say "!!! DATA_ROOT 에 eval/ 없음 → bash run.sh data 먼저"; exit 1; }   # 평가는 교사 라벨도 풀 이미지도 쓰지 않는다 (어댑터 + 평가셋만)
else
  [ -n "$LABELS" ] || { say "!!! 교사 라벨을 찾지 못했다. 찾아본 곳: $LABEL_TRIED"; say "!!! → bash run.sh label $POOL 을 먼저 요청할 것"; exit 1; }
  [ -f "$PCFG" ] || { say "!!! 풀 설정 파일 없음: $PCFG"; exit 1; }
  [ -d "$DATA_ROOT/pool" ] || { say "!!! DATA_ROOT 에 pool/ 없음 → bash run.sh data 먼저"; exit 1; }
  { [ "$MODE" = train ] || [ -d "$DATA_ROOT/eval" ]; } || { say "!!! DATA_ROOT 에 eval/ 없음 → bash run.sh data 먼저"; exit 1; }
fi
CELLS=${CELLS:-$(python -c "import json;print(' '.join(c['tag'] for c in json.load(open('$ARMS'))['cells']))")}
say "[$MODE] 풀 $POOL 조건 $COND 라벨 $LABELS 셀: $CELLS"
if [ "$MODE" != eval ]; then   # 라벨 커버리지 확인 (평가는 라벨을 쓰지 않으므로 건너뜀)
CRC=0; python - "$LABELS" "$POOL" > "$OUT_ROOT/coverage_${POOL}.txt" 2>&1 <<'PYC' || CRC=$?
import sys, glob, pandas as pd; lab, pool = sys.argv[1:3]
L = pd.read_parquet(lab)[["image_id", "pixel_index"]].drop_duplicates(); need = pd.concat([pd.read_parquet(p) for p in glob.glob(f"pools/{pool}/rows_*.parquet")]).drop_duplicates()
m = need.merge(L, on=["image_id", "pixel_index"], how="left", indicator=True); miss = int((m._merge == "left_only").sum())
print(f"[grid] 라벨 커버리지: 필요 {len(need)} px, 부족 {miss} px"); sys.exit(1 if miss else 0)
PYC
cat "$OUT_ROOT/coverage_${POOL}.txt" | tee -a "$LOG"; rm -f "$OUT_ROOT/coverage_${POOL}.txt"; [ "$CRC" = 0 ] || { say "!!! 라벨 부족 → bash run.sh label $POOL 먼저"; exit 1; }
fi
train_cell() { local cell=$1; local ad=$OUT_ROOT/checkpoints/${COND}_${cell}${SUFFIX}
  [ -f "$ad/adapter_model.safetensors" ] && { say "$cell 학습 완료됨 — 건너뜀"; return 0; }
  say "학습 $COND $cell"; python -u experiments/20_train_student.py --cond "$COND" --epochs 2 --accum 8 --focal "$FOCAL" --labels "$LABELS" --pools "$PCFG" --rows "pools/$POOL/rows_$cell.parquet" --tag "_${cell}${SUFFIX}" > "$OUT_ROOT/train_${COND}_${cell}${SUFFIX}.log" 2>&1 || say "!!! 학습 실패 $cell"; }
eval_cell() { local cell=$1; local ad=$OUT_ROOT/checkpoints/${COND}_${cell}${SUFFIX}; [ -f "$ad/adapter_model.safetensors" ] || return 0; [ -n "$EVAL_DATASETS" ] || return 0
  for es in $EVAL_SETS; do [ "$es" = none ] && continue; local suf="" todo="" ds; [ "$es" = large ] && suf=_large
    local base=$OUT_ROOT/eval/eval_${COND}_${cell}${SUFFIX}${suf}   # 데이터셋별 파일 <base>__<ds>.parquet, 예전 한 파일 <base>.parquet 는 ibims1·nyuv2·eth3d 를 담는다
    for ds in $EVAL_DATASETS; do [ -f "${base}__${ds}.parquet" ] && continue; [ -f "${base}.parquet" ] && case $ds in ibims1|nyuv2|eth3d) continue;; esac; todo="$todo,$ds"; done
    [ -n "$todo" ] || continue
    say "평가 $COND $cell ($es: ${todo#,})"; python -u experiments/21_eval_student.py --tag "${COND}_${cell}${SUFFIX}" --adapter "$ad" --focal "$FOCAL" --eval_set "$es" --datasets "${todo#,}" ${EVAL_EXTRA:-} >> "$OUT_ROOT/eval_${COND}_${cell}${SUFFIX}_${es}.log" 2>&1 || say "!!! 평가 실패 $cell $es"; done; }
run_cells() { local fn=$1; if [ "$NPROC_TRAIN" -gt 1 ]; then   # GPU 한 장 통째: 셀 NPROC_TRAIN 개를 같은 GPU 에서 동시에 (학습 ≈10 GB, 평가 ≈8 GB)
    local i=0; for cell in $CELLS; do if [ "$NDEV" -gt 1 ]; then CUDA_VISIBLE_DEVICES=${DEVS[$((i % NDEV))]} $fn "$cell" & else $fn "$cell" & fi; i=$((i+1)); [ $((i % NPROC_TRAIN)) -eq 0 ] && wait; done; wait
  else for cell in $CELLS; do $fn "$cell"; done; fi; }
NCELL=$(echo $CELLS | wc -w)
if [ "$MODE" != train ]; then   # 주행 평가 세트(DDAD·nuScenes)는 30 GB 아카이브 밖의 별도 묶음(drive_eval, 550 MB)이다. /app/data 아래에 풀려 있거나 zip 으로 있으면 $DATA_ROOT/eval 에 연결한다
  for ds in $EVAL_DATASETS; do case $ds in ddad|nuscenes) ;; *) continue;; esac; [ -f "$DATA_ROOT/eval/$ds/${ds}_val.jsonl" ] && continue
    DE=""; for d in /app/data "$DATA_SRC" "$WORK_ROOT/drive_eval"; do [ -d "$d" ] && [ -z "$DE" ] && DE=$(find -L "$d" -maxdepth 5 -name "${ds}_val.jsonl" -path "*/$ds/*" -printf "%h\n" 2>/dev/null | head -1 || true); done
    if [ -z "$DE" ]; then Z=$(python - "$ds" /app/data "$DATA_SRC" <<'PYF'
import sys, os, zipfile; ds = sys.argv[1]
for root in sys.argv[2:]:
    for dp, dn, fn in os.walk(root, followlinks=True):
        if dp[len(root):].count(os.sep) >= 4: dn[:] = []
        for f in fn:
            p = os.path.join(dp, f)
            if f.endswith(".zip") and zipfile.is_zipfile(p) and any(n.endswith(f"{ds}/{ds}_val.jsonl") for n in zipfile.ZipFile(p).namelist()): print(p); sys.exit(0)
PYF
)
      if [ -n "$Z" ]; then if python - "$Z" "$WORK_ROOT/drive_eval" <<'PYE'
import sys, zipfile, hashlib, os; z, out = sys.argv[1:3]; zipfile.ZipFile(z).extractall(out)
for dp, _, fn in os.walk(out):   # 묶음에 든 SHA256SUMS 로 검증 (경로는 SHA256SUMS 가 있는 폴더 기준)
    if "SHA256SUMS" in fn:
        bad = [l.split()[1] for l in open(f"{dp}/SHA256SUMS") if hashlib.sha256(open(f"{dp}/{l.split()[1]}", "rb").read()).hexdigest() != l.split()[0]]
        print(f"[eval] drive_eval 풀기 {z} → {out}, SHA256 불일치 {len(bad)} 개"); sys.exit(1 if bad else 0)
print(f"[eval] drive_eval 풀기 {z} → {out} (SHA256SUMS 없음 — 검증 생략)")
PYE
      then DE=$(find "$WORK_ROOT/drive_eval" -maxdepth 4 -name "${ds}_val.jsonl" -printf "%h\n" | head -1); else say "!!! [eval] $ds: $Z 풀기 또는 SHA256 검증 실패"; fi; fi
    fi
    if [ -n "$DE" ]; then mkdir -p "$DATA_ROOT/eval" && ln -sfn "$DE" "$DATA_ROOT/eval/$ds" && say "[eval] $ds: $DE 를 $DATA_ROOT/eval/$ds 로 연결 ($(wc -l < "$DE/${ds}_val.jsonl") 장)" || say "!!! [eval] $ds: $DATA_ROOT/eval 에 연결 실패 (쓰기 권한)"; fi; done
fi
if [ "$MODE" != train ]; then   # 주행 평가 세트(DDAD·nuScenes)는 아카이브에 없다 → 관리자가 /app/data 에 넣어 준 depthlm_drive_eval.tar 를 찾아 작업 공간에 풀고 연결한다
  MISS=""; for ds in $EVAL_DATASETS; do [ -f "$DATA_ROOT/eval/$ds/${ds}_val.jsonl" ] || MISS="$MISS $ds"; done
  if [ -n "$MISS" ]; then
    DT=""; for d in /app/data "${DATA_SRC:-}"; do [ -n "$d" ] && [ -d "$d" ] && [ -z "$DT" ] && DT=$(find "$d" -maxdepth 4 -name "depthlm_drive_eval.tar" 2>/dev/null | head -1 || true); done
    if [ -n "$DT" ]; then
      DX=$WORK_ROOT/drive_eval; mkdir -p "$DX"
      if [ -f "$DT.sha256" ] && ! (cd "$(dirname "$DT")" && sha256sum -c "$(basename "$DT").sha256" >/dev/null 2>&1); then say "!!! [eval] $DT 체크섬 불일치 — 주행 세트를 쓰지 않는다"
      else
        tar -xf "$DT" -C "$DX" && for ds in $MISS; do [ -f "$DX/$ds/${ds}_val.jsonl" ] && ln -sfn "$DX/$ds" "$DATA_ROOT/eval/$ds" 2>/dev/null && say "[eval] 주행 평가 세트 연결: $ds ← $DT"; done
      fi
    fi
  fi
fi
if [ "$MODE" != train ]; then   # 이 환경에 없는 평가 세트는 건너뛴다 (예: H200 아카이브에는 주행 세트가 없다 → 어댑터를 zip 으로 받아 로컬에서 평가)
  AV=""; for ds in $EVAL_DATASETS; do if [ -f "$DATA_ROOT/eval/$ds/${ds}_val.jsonl" ]; then AV="$AV $ds"; else say "[eval] $ds: $DATA_ROOT/eval/$ds 에 평가 세트가 없어 이 작업에서는 건너뜀 (어댑터로 로컬에서 평가)"; fi; done; EVAL_DATASETS=${AV# }
  say "[eval] 평가 세트: ${EVAL_DATASETS:-없음 → 학습과 결과 zip(어댑터 포함)만} | 세트 종류: $EVAL_SETS"; fi
case $MODE in
  train) run_cells train_cell
         say "[train] 완료 $POOL $COND — 셀별 어댑터:"
         NOK=0; for c in $CELLS; do if [ -f "$OUT_ROOT/checkpoints/${COND}_${c}${SUFFIX}/adapter_model.safetensors" ]; then NOK=$((NOK+1)); say "  $c  OK"; else say "  $c  !!! 실패 → $OUT_ROOT/train_${COND}_${c}${SUFFIX}.log 확인"; fi; done
         say "[train] $NOK/$NCELL 셀 성공. 어댑터: $OUT_ROOT/checkpoints/${COND}_*${SUFFIX}"
         say "[train] 다음 단계는 bash run.sh eval $POOL $COND — 단, 어댑터가 $OUT_ROOT 에 남아 있어야 한다"
         say "[train] 결과 볼륨이 작업 사이에 비워지는 환경(사업단 H200, 2026-09-28 실측)이면 eval 이 어댑터를 못 찾는다. 그 경우 학습과 평가를 한 작업으로 도는 bash run.sh grid $POOL $COND 를 쓸 것"
         [ "$NOK" -gt 0 ] || exit 1
         exit 0;;
  eval)  NAD=0; for c in $CELLS; do [ -f "$OUT_ROOT/checkpoints/${COND}_${c}${SUFFIX}/adapter_model.safetensors" ] && NAD=$((NAD+1)) || true; done
         [ "$NAD" -gt 0 ] || { say "!!! [eval] $OUT_ROOT/checkpoints 에 ${COND}_*${SUFFIX} 어댑터가 없다"
           say "!!! → 같은 파드에서 학습했다면 bash run.sh train $POOL $COND 먼저"
           say "!!! → 결과 볼륨이 작업 사이에 비워지는 환경이면 학습 결과가 넘어오지 않는다. bash run.sh grid $POOL $COND 로 학습과 평가를 한 작업에 돌릴 것"; exit 1; }
         say "[eval] 학습된 어댑터 $NAD/$NCELL 셀"; run_cells eval_cell;;
  *)     run_cells train_cell; run_cells eval_cell;;
esac
for es in $EVAL_SETS; do [ "$es" = none ] || [ -z "$EVAL_DATASETS" ] && continue; python experiments/31_grid.py --cond "$COND" --suffix "$SUFFIX" --eval_set "$es" --arms "$ARMS" --datasets "$(echo $EVAL_DATASETS | tr ' ' ',')" >> "$LOG" 2>&1 || say "!!! 표 실패 $es"; done
say "[$MODE] 완료 $POOL $COND. 결과: $OUT_ROOT/{eval,tables,figures,checkpoints}"
for es in $EVAL_SETS; do [ "$es" = none ] || [ -z "$EVAL_DATASETS" ] && continue; suf=""; [ "$es" = large ] && suf=_large; echo "===== 요약 $POOL $COND ($es) ====="; head -40 "$OUT_ROOT/tables/table_grid_${COND}${SUFFIX}${suf}.md" 2>/dev/null || echo "  (표 파일 없음: tables/table_grid_${COND}${SUFFIX}${suf}.md — 31_grid.py 로그 확인)"; done
python - "$OUT_ROOT" "results_${COND}_${POOL}" "${COND}_" "$SUFFIX" <<'PYS' || say "!!! zip 생성 실패 — 결과 파일은 $OUT_ROOT 에 그대로 있다 (여유 공간 확인)"
import sys, os, zipfile; root, name, cond, suffix = sys.argv[1:5]
with zipfile.ZipFile(f"{root}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("hf", "data", "labels")]
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), root)
            if (cond in rel and suffix in rel) or rel.startswith("run_"): z.write(os.path.join(dp, f), rel)
print(f"zip: {root}/{name}.zip  {os.path.getsize(f'{root}/{name}.zip')/1e6:.1f} MB")
PYS
