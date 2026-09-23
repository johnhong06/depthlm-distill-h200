#!/usr/bin/env bash
# 데이터(풀 이미지 2.7 GB + 평가셋 0.4 GB)를 GitHub Release 자산에서 받아 DATA_ROOT 에 풀기. 사용: DATA_URL_BASE=<release 자산 URL 접두사> bash scripts/fetch_data.sh
set -euo pipefail; cd "$(dirname "$0")/.."; export DATA_ROOT=${DATA_ROOT:-$PWD/data}; mkdir -p "$DATA_ROOT"
[ -d "$DATA_ROOT/pool" ] && [ -d "$DATA_ROOT/eval" ] && { echo "데이터 이미 있음: $DATA_ROOT"; exit 0; }
BASE=${DATA_URL_BASE:?"DATA_URL_BASE 필요 (예: https://github.com/<user>/<repo>/releases/download/data-v3)"}
PARTS=${DATA_PARTS:-"aa ab ac ad"}; TMP=$(mktemp -d)
for p in $PARTS; do echo "GET part_$p"; curl -L --retry 5 -o "$TMP/depthlm_distill_data.tar.part_$p" "$BASE/depthlm_distill_data.tar.part_$p"; done
curl -L --retry 5 -o "$TMP/SHA256SUMS" "$BASE/SHA256SUMS" && (cd "$TMP" && sha256sum -c SHA256SUMS)
cat "$TMP"/depthlm_distill_data.tar.part_* | tar -xf - -C "$DATA_ROOT" --strip-components=1 && rm -rf "$TMP"
echo "완료: $(find "$DATA_ROOT/pool" -type f | wc -l) 풀 이미지, $(find "$DATA_ROOT/eval" -type f | wc -l) 평가 파일"
