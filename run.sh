#!/usr/bin/env bash
# DepthLM 증류 실험 — 컨테이너 진입점 (비대화형). 사용법:
#   MODE=smoke                       bash run.sh        # 파이프라인 검증 (모델 다운로드 → 30 스텝 학습 → 3 px 평가)
#   MODE=grid POOL=mixed COND=soft   bash run.sh        # 격자 8셀 학습 → 평가(small+large) → 표·그림   (COND = soft | hard)
# 환경변수: DATA_ROOT(데이터 루트, 기본 ./data), OUT_ROOT(결과, 기본 ./results), FOCAL(750), CELLS(기본 arms.json 전체), EVAL_SETS("small large"),
#           HF_TOKEN(gated 모델용), NPROC(병렬 학습 프로세스 수, MIG 슬라이스 격리 시 CUDA_VISIBLE_DEVICES 로 분배)
set -euo pipefail; cd "$(dirname "$0")"; export PYTHONUNBUFFERED=1
EXTRACTED_HERE=0   # 이 작업이 $OUT_ROOT/data 에 데이터를 풀었으면 1 → 끝날 때 지워서 결과만 남긴다 (관리자 서버에 30 GB 가 작업마다 쌓이지 않게)
cleanup() { if [ "${H200_CHILD:-0}" = 0 ] && [ "${KEEP_DATA:-0}" = 0 ]; then
    [ "$EXTRACTED_HERE" = 1 ] && rm -rf "$OUT_ROOT/data" && echo "[cleanup] 작업용 데이터 복사본 삭제 ($OUT_ROOT/data)"
    [ -d /app/output ] && [ -d "$OUT_ROOT/hf" ] && rm -rf "$OUT_ROOT/hf" && echo "[cleanup] 모델 캐시 삭제 ($OUT_ROOT/hf) — 결과(체크포인트·평가·표·zip·라벨)만 남음"; fi; return 0; }
trap cleanup EXIT
# 위치 인자: bash run.sh <smoke|label|grid|all> [pool] [cond] [hf_token]   (환경변수 MODE/POOL/COND/HF_TOKEN 도 동일하게 동작; 토큰은 /app/data/hf_token.txt 로도 가능)
# all = 혼합 격자 2개 → 실내·실외 라벨링(라벨이 없을 때만) → 실내·실외 격자 4개를 한 작업으로 이어서 실행
ARGS=(); for a in "$@"; do case $a in hf_*) export HF_TOKEN=$a;; *) ARGS+=("$a");; esac; done   # hf_ 로 시작하는 인자는 위치와 무관하게 토큰
MODE=${ARGS[0]:-${MODE:-smoke}}; POOL=${ARGS[1]:-${POOL:-mixed}}; COND=${ARGS[2]:-${COND:-soft}}
FOCAL=${FOCAL:-750}; EVAL_SETS=${EVAL_SETS:-"small large"}; export HF_HUB_DISABLE_PROGRESS_BARS=1
# 사업단 파드 규격: 데이터는 /app/data (읽기), 결과는 /app/output (파드 종료 후 보존). 없으면 로컬 기본값.
export DATA_ROOT=${DATA_ROOT:-$([ -d /app/data/depthlm_distill_h200 ] && echo /app/data/depthlm_distill_h200 || { [ -d /app/data ] && echo /app/data || echo $PWD/data; })}   # 관리자가 tar 를 푼 폴더 우선
export OUT_ROOT=${OUT_ROOT:-$([ -d /app/output ] && echo /app/output || echo $PWD/results)}; mkdir -p "$OUT_ROOT"
DATA_SRC=$DATA_ROOT   # 팩 파일이 놓인 원래 위치 (풀린 뒤 DATA_ROOT 가 바뀌어도 추가 팩은 여기서 찾는다)
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
# 관리자 부담 최소화: 드라이브의 조각(depthlm_distill_h200_app_data.tar.part_*)을 /app/data 아래 아무 폴더에 받아 두기만 하면 스크립트가 검증하고 /app/output/data 에 한 번 푼다
if [ ! -d "$DATA_ROOT/pool" ]; then
  PRE=""; for d in "$OUT_ROOT/data" /app/data "$DATA_SRC"; do [ -d "$d" ] && [ -z "$PRE" ] && PRE=$(find "$d" -maxdepth 4 -type d -name "depthlm_distill_h200" 2>/dev/null | head -1 || true); done
  if [ -n "$PRE" ] && [ -d "$PRE/pool" ]; then export DATA_ROOT=$PRE; echo "[data] 이미 풀린 폴더 사용: $DATA_ROOT"
  else
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
        ZD=$(dirname "$(echo $ZIPS | cut -d' ' -f1)"); if touch "$ZD/.write_test" 2>/dev/null; then rm -f "$ZD/.write_test"; XDIR=$ZD; echo "[data] zip 발견: $ZIPS → 폴더가 쓰기 가능, 제자리에서 풀기"; else XDIR=$OUT_ROOT/data; echo "[data] zip 발견: $ZIPS → 읽기 전용, $OUT_ROOT/data 에 풀기 (30 GB)"; fi
        mkdir -p "$XDIR"; python - "$XDIR" $ZIPS <<'PYZ' && export DATA_ROOT=$XDIR/depthlm_distill_h200 && { [ "$XDIR" = "$OUT_ROOT/data" ] && EXTRACTED_HERE=1 || true; } && echo "[data] 풀기 완료: 풀 이미지 $(find "$DATA_ROOT/pool" -type f | wc -l), 평가 파일 $(find "$DATA_ROOT/eval" -type f | wc -l), 교사 가중치 조각 $(ls "$DATA_ROOT/models/DepthLM" | grep -c safetensors)" || { echo "!!! [data] zip 에서 풀기 실패 (조각 누락 또는 SHA256 불일치)"; exit 1; }
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
      if touch "$PDIR/.write_test" 2>/dev/null; then rm -f "$PDIR/.write_test"; XDIR=$PDIR; echo "[data] 조각 발견: $PDIR (16 개). 폴더가 쓰기 가능 → 제자리에서 풀기 (복사본 없음)"
      else XDIR=$OUT_ROOT/data; echo "[data] 조각 발견: $PDIR (16 개). 폴더가 읽기 전용 → $OUT_ROOT/data 에 풀기 (30 GB)"; fi
      (cd "$PDIR" && sha256sum -c --quiet SHA256SUMS_parts) && echo "[data] 조각 SHA256 검증 통과" || { echo "!!! [data] 조각 SHA256 불일치 — 업로드가 덜 됐거나 깨짐. 다시 받을 것"; exit 1; }
      TMPX=$XDIR/.extracting_$$; rm -rf "$TMPX"; mkdir -p "$TMPX"   # 임시 폴더에 풀고 성공했을 때만 최종 이름으로 (중간에 죽어도 반쪽짜리 폴더가 남지 않음)
      if cat "$PDIR"/depthlm_distill_h200_app_data.tar.part_* | tar -xf - -C "$TMPX" && mv "$TMPX/depthlm_distill_h200" "$XDIR/depthlm_distill_h200"; then rm -rf "$TMPX"; export DATA_ROOT=$XDIR/depthlm_distill_h200; [ "$XDIR" = "$OUT_ROOT/data" ] && EXTRACTED_HERE=1; echo "[data] 풀기 완료: 풀 이미지 $(find "$DATA_ROOT/pool" -type f | wc -l), 평가 파일 $(find "$DATA_ROOT/eval" -type f | wc -l), 교사 가중치 조각 $(ls "$DATA_ROOT/models/DepthLM" | grep -c safetensors)"
      else rm -rf "$TMPX"; echo "!!! [data] 풀기 실패 — 임시 폴더 정리함. 다시 요청할 것"; exit 1; fi
    fi
  fi
fi
# 데이터 확보 순서: ① /app/data 에 풀려 있음 → ② 이전 작업이 /app/output/data 에 풀어 둠 → ③ /app/data 의 tar 분할본 → ④ HF 비공개 데이터셋(DATA_REPO)에서 토큰으로 내려받음
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
  if [ -d "$OUT_ROOT/data/pool" ]; then export DATA_ROOT=$OUT_ROOT/data
  else
    PACK=""; for d in "$DATA_ROOT" /app/data "$DATA_SRC"; do [ -d "$d" ] && [ -z "$PACK" ] && PACK=$(find "$d" -maxdepth 3 -name "depthlm_distill_data.tar.part_aa" -printf "%h\n" 2>/dev/null | head -1 || true); done
    if [ -z "$PACK" ] && [ -n "${HF_TOKEN:-}" ]; then
      echo "[data] $DATA_REPO 에서 데이터 팩 다운로드 (6 GB)"; mkdir -p "$OUT_ROOT/data_pack"
      hf_fetch "$OUT_ROOT/data_pack" "depthlm_distill_data.tar.part_*" "SHA256SUMS" && PACK=$OUT_ROOT/data_pack || echo "!!! [data] 다운로드 실패 — 토큰이 $DATA_REPO 를 읽을 수 있는지 확인"
    fi
    if [ -n "$PACK" ]; then
      [ -f "$PACK/SHA256SUMS" ] && { (cd "$PACK" && sha256sum -c --quiet SHA256SUMS) && echo "[data] SHA256 검증 통과" || { echo "!!! [data] SHA256 불일치 — 분할본이 깨짐"; exit 1; }; }
      mkdir -p "$OUT_ROOT/data"; cat "$PACK"/depthlm_distill_data.tar.part_* | tar -xf - -C "$OUT_ROOT/data" --strip-components=1 && export DATA_ROOT=$OUT_ROOT/data
      [ "$PACK" = "$OUT_ROOT/data_pack" ] && rm -rf "$OUT_ROOT/data_pack"; echo "[data] 풀기 완료: $(find "$DATA_ROOT/pool" -type f | wc -l) 풀 이미지, $(find "$DATA_ROOT/eval" -type f | wc -l) 평가 파일"
    fi
  fi
fi
if [ -d "$DATA_ROOT/pool" ] && [ ! -f "$DATA_ROOT/.extra_done" ] && [ ! -f "$DATA_ROOT/extra_done.txt" ]; then   # 추가 팩 (실내 v4 새 이미지). 관리자 묶음에는 이미 포함(extra_done.txt)
  EX=""; for d in /app/data "$DATA_SRC" "$DATA_ROOT"; do [ -d "$d" ] && [ -z "$EX" ] && EX=$(find "$d" -maxdepth 3 -name "depthlm_distill_data_extra.tar" -printf "%h\n" 2>/dev/null | head -1 || true); done
  if [ -z "$EX" ] && [ -n "${HF_TOKEN:-}" ]; then mkdir -p "$OUT_ROOT/data_pack"; hf_fetch "$OUT_ROOT/data_pack" "depthlm_distill_data_extra.tar" "SHA256SUMS_extra" >/dev/null 2>&1 || true; [ -f "$OUT_ROOT/data_pack/depthlm_distill_data_extra.tar" ] && EX=$OUT_ROOT/data_pack; fi
  if [ -n "$EX" ] && [ -w "$DATA_ROOT" ]; then
    [ -f "$EX/SHA256SUMS_extra" ] && { (cd "$EX" && sha256sum -c --quiet SHA256SUMS_extra) || { echo "!!! [data] 추가 팩 SHA256 불일치"; exit 1; }; }
    tar -xf "$EX/depthlm_distill_data_extra.tar" -C "$DATA_ROOT" --strip-components=1 && touch "$DATA_ROOT/.extra_done" && echo "[data] 추가 팩 풀기 완료 ($(tar -tf "$EX/depthlm_distill_data_extra.tar" | grep -cE '\.(png|jpg)$') 장)"
    [ "$EX" = "$OUT_ROOT/data_pack" ] && rm -rf "$OUT_ROOT/data_pack"
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
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG" || say "nvidia-smi 없음"
GPU_MB=$(python -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/2**20) if torch.cuda.is_available() else 0)")   # nvidia-smi 는 컨테이너에서 메모리 값을 못 줄 수 있어 torch 로 판정
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
if [ "$MODE" = all ]; then   # 한 이슈로 전체 체인. 각 단계는 하위 실행이라 하나가 실패해도 다음으로 넘어간다. 라벨링은 빠진 쌍만 한다
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
  say "[label] 완료: $LD/teacher_labels.parquet"; exit 0
fi
# --- grid ---  태그 = <cond>_<cell>_<pool>_f<focal>  (풀이 달라도 체크포인트·평가 파일이 겹치지 않음)
SUFFIX=_${POOL}_f${FOCAL}; ARMS=pools/$POOL/arms.json; PCFG=configs/pool_${POOL}.yaml
LABELS=$OUT_ROOT/labels/$POOL/teacher_labels.parquet; [ -f "$LABELS" ] || LABELS=pools/$POOL/teacher_labels.parquet   # 파드에서 병합한 완전본 우선, 없으면 저장소 라벨
[ -f "$ARMS" ] && [ -f "$LABELS" ] && [ -f "$PCFG" ] || { say "!!! 풀 파일 없음: $ARMS $LABELS $PCFG"; exit 1; }
[ -d "$DATA_ROOT/pool" ] && [ -d "$DATA_ROOT/eval" ] || { say "!!! DATA_ROOT 에 pool/ eval/ 없음 → scripts/fetch_data.sh 먼저"; exit 1; }
CELLS=${CELLS:-$(python -c "import json;print(' '.join(c['tag'] for c in json.load(open('$ARMS'))['cells']))")}
say "[grid] 풀 $POOL 조건 $COND 라벨 $LABELS 셀: $CELLS"
CRC=0; python - "$LABELS" "$POOL" > "$OUT_ROOT/coverage_${POOL}.txt" 2>&1 <<'PYC' || CRC=$?
import sys, glob, pandas as pd; lab, pool = sys.argv[1:3]
L = pd.read_parquet(lab)[["image_id", "pixel_index"]].drop_duplicates(); need = pd.concat([pd.read_parquet(p) for p in glob.glob(f"pools/{pool}/rows_*.parquet")]).drop_duplicates()
m = need.merge(L, on=["image_id", "pixel_index"], how="left", indicator=True); miss = int((m._merge == "left_only").sum())
print(f"[grid] 라벨 커버리지: 필요 {len(need)} px, 부족 {miss} px"); sys.exit(1 if miss else 0)
PYC
cat "$OUT_ROOT/coverage_${POOL}.txt" | tee -a "$LOG"; rm -f "$OUT_ROOT/coverage_${POOL}.txt"; [ "$CRC" = 0 ] || { say "!!! [grid] 라벨 부족 → bash run.sh label $POOL 먼저 (all 모드는 자동)"; exit 1; }
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
