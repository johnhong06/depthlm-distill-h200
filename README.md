# depthlm-distill-h200

DepthLM(12B) → Qwen2.5-VL-3B 증류 실험을 비대화형 컨테이너(사업단 H200, GitHub Issue → Jenkins)에서 돌리기 위한 저장소.
교사 라벨링은 로컬에서 끝낸 결과(`pools/*/teacher_labels.parquet`)를 포함하므로 컨테이너에서는 **학생 학습·평가만** 돌린다 (VRAM ≈ 10 GB, MIG 16 GB 슬라이스로 충분).

## 실행 명령 (이슈 양식의 "실행 명령어")

```bash
# 0) 스모크: 파이프라인 검증 (모델 다운로드 → 30 스텝 학습 → 3 px 평가, GPU 5 분 내외)
HF_TOKEN=<토큰> MODE=smoke bash run.sh

# 1) 데이터 준비 (풀 이미지 2.7 GB + 평가셋 0.4 GB, GitHub Release 자산에서 다운로드) — grid 전에 한 번
DATA_URL_BASE=https://github.com/<user>/depthlm-distill-h200/releases/download/data-v3 bash scripts/fetch_data.sh

# 2) 격자 실험 하나 (풀 × 손실): 8셀 학습 → 소형+대형 평가 → 표·그림. 결과는 results/ 에 모임
HF_TOKEN=<토큰> MODE=grid POOL=mixed COND=soft bash run.sh
```

`run.sh` 는 재개 가능하다(완료된 셀·평가는 건너뜀). 한 작업에 여러 GPU(슬라이스)가 보이면 `NPROC=<개수>` 로 셀을 병렬 학습한다.

## 실험 목록 (6 작업)

| 작업 | POOL | COND | 내용 | 상태 |
|---|---|---|---|---|
| 1 | indoor | soft | 실내 풀, 분포(KL) 증류, 8셀 | 풀 준비 중 (로컬 라벨링 후 추가) |
| 2 | outdoor | soft | 주행 풀, 분포 증류, 8셀 | 풀 준비 중 |
| 3 | mixed | soft | 실내 50 / 주행 50 풀, 분포 증류, 8셀 | 포함 (로컬에서도 진행 중) |
| 4 | indoor | hard | 실내 풀, greedy(CE) 증류 | 풀 준비 중 |
| 5 | outdoor | hard | 주행 풀, greedy 증류 | 풀 준비 중 |
| 6 | mixed | hard | 혼합 풀, greedy 증류 | 포함 |

격자 8셀: 이미지 수 N ∈ {400, 1600, 6400} × 이미지당 픽셀 k ∈ {1, 4, 16}, N·k ≤ 25,600. 셀당 2 epoch, LoRA r16, 학생 입력 초점 750.
예상 시간(RTX PRO 4500 기준 0.46 s/step): 작업당 학습 ≈ 19 h + 평가 ≈ 9 h. H200 은 이보다 짧을 것으로 예상(실측 필요).

## 환경

- Python ≥ 3.10, `pip install -r requirements.txt` (torch 는 CUDA 에 맞는 빌드로; transformers 5.16.1, peft ≥ 0.20 고정)
- 모델: `Qwen/Qwen2.5-VL-3B-Instruct` (Apache-2.0, 7.5 GB, 자동 다운로드). 교사 `facebook/DepthLM` 은 이 저장소의 학습·평가에서는 **불필요** (라벨이 포함됨).
- 컨테이너 안에서 Hugging Face 접근이 필요하며, gated 모델은 쓰지 않으므로 HF_TOKEN 은 없어도 된다 (rate limit 회피용으로만).
- 데이터: `data/pool/…` (풀 이미지), `data/eval/{ibims1,nyuv2,eth3d}/…` (평가). `DATA_ROOT` 로 위치 변경 가능.

## 결과 (results/)

| 경로 | 내용 |
|---|---|
| `results/checkpoints/<cond>_<cell>_f750/` | LoRA 어댑터 (`adapter_model.safetensors`, `lora_adapter.pt`, `train.log`) |
| `results/eval/eval_<cond>_<cell>_f750[_large].parquet` | 픽셀별 예측·GT·불확실도 |
| `results/tables/table_grid_<cond>_f750[_large].md` | 셀별 δ1·CI, 행·열·고정예산 쌍대 비교 |
| `results/figures/fig_grid_<cond>_f750[_large].png` | 예산 대 δ1 |
| `results/run_*.log`, `results/train_*.log`, `results/eval_*.log` | 로그 |

## 공정 비교 규칙

같은 풀 안의 모든 셀·손실은 같은 픽셀 목록·같은 교사 라벨·같은 스텝·초기화·셔플 시드를 쓴다. 셀 간 차이는 N·k 뿐이고, 손실 간 차이는 학습 목표(교사 숫자 CE vs 교사 자릿수 분포 KL)뿐이다. ETH3D 는 완전 held-out.

## 라이선스

코드 MIT. DepthLM 은 FAIR Noncommercial Research License(교사 라벨 생성에 사용, 비상업 연구 목적). 데이터셋(SUN RGB-D, NYUv2, KITTI, iBims-1, ETH3D)은 각 원 라이선스(비상업 연구)를 따르며 재배포하지 않는다.
