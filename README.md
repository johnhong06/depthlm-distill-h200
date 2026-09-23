# depthlm-distill-h200

DepthLM(12B) → Qwen2.5-VL-3B 증류 실험을 비대화형 컨테이너(사업단 H200, GitHub Issue → Jenkins)에서 돌리기 위한 저장소.
교사 라벨링은 로컬에서 끝낸 결과(`pools/*/teacher_labels.parquet`)를 포함하므로 컨테이너에서는 **학생 학습·평가만** 돌린다 (VRAM ≈ 10 GB, MIG 16 GB 슬라이스로 충분).

## 사업단 파드 규격에 맞춘 실행

- 데이터는 사업단이 `/app/data/` 에 넣어 준다 (아래 "데이터 전달" 레이아웃). 결과는 `/app/output/` 에 쓰며 파드 종료 후 보존된다.
- `run.sh` 는 `/app/data`, `/app/output` 이 있으면 자동으로 그 경로를 쓴다. 파드 실행 중 코드 수정은 반영되지 않으므로 필요한 값은 모두 실행 명령의 환경변수로 준다.
- 인터넷은 허용됨: 학생 모델 `Qwen/Qwen2.5-VL-3B-Instruct`(Apache-2.0, 7.5 GB)는 실행 중 내려받는다. 교사 모델(gated)은 라벨링 작업에서만 필요하며 `HF_TOKEN=<읽기전용 토큰>` 을 명령에 넣는다. `/app/data/models/` 에 가중치를 두면 다운로드 없이 그것을 쓴다.

| 작업 | 실행 명령어 | MIG | 예상 시간 |
|---|---|---|---|
| 스모크 | `MODE=smoke bash run.sh` | 1 | 10 분 |
| 교사 라벨링 (실내 풀) | `HF_TOKEN=<토큰> MODE=label POOL=indoor bash run.sh` | **7** (VRAM 30 GB) | 44,800 px, RTX 기준 12 h |
| 교사 라벨링 (실외 풀) | `HF_TOKEN=<토큰> MODE=label POOL=outdoor bash run.sh` | **7** | 12 h |
| 격자 (풀 × 손실) 6개 | `MODE=grid POOL=<indoor|outdoor|mixed> COND=<soft|hard> bash run.sh` | 1 | 작업당 학습 19 h + 평가 9 h (RTX 기준) |

라벨링 결과 `/app/output/labels/<POOL>/teacher_labels.parquet` 를 돌려받아 `pools/<POOL>/` 에 넣고 커밋한 뒤 격자 작업을 낸다. 혼합(mixed) 풀은 라벨이 이미 포함되어 있어 바로 격자 작업이 가능하다.
`run.sh` 는 재개 가능하다(완료된 셀·평가는 건너뜀). MIG 7 로 격자 작업을 내면 `NPROC=7` 로 셀을 병렬 학습할 수 있다.

## 데이터 전달 (사업단 `/app/data/`)

tar 분할본 `depthlm_distill_data.tar.part_*`(총 약 5.8 GB, 풀 3개 이미지 + 평가셋)을 `/app/data/` 에 그대로 두면 `run.sh` 가 첫 실행 때 `/app/output/data/` 에 풀어 쓴다. 미리 풀어 두는 경우의 레이아웃:

```
/app/data/pool/...   # 풀 이미지 (혼합·실내·실외 풀이 공유; sunrgbd/, kitti/, distill_pool/ 하위)
/app/data/eval/ibims1/, nyuv2/, eth3d/   # 평가 이미지 + jsonl
/app/data/models/Qwen2.5-VL-3B-Instruct/  # (선택) 학생 가중치, 없으면 다운로드
```

## 실험 목록 (6 작업)

| 작업 | POOL | COND | 내용 | 상태 |
|---|---|---|---|---|
| 1 | indoor | soft | 실내 풀, 분포(KL) 증류, 8셀 | 풀 포함, 라벨링 필요 (MODE=label) |
| 2 | outdoor | soft | 주행 풀, 분포 증류, 8셀 | 풀 포함, 라벨링 필요 |
| 3 | mixed | soft | 실내 50 / 주행 50 풀, 분포 증류, 8셀 | 포함 (로컬에서도 진행 중) |
| 4 | indoor | hard | 실내 풀, greedy(CE) 증류 | 풀 포함, 라벨링 필요 |
| 5 | outdoor | hard | 주행 풀, greedy 증류 | 풀 포함, 라벨링 필요 |
| 6 | mixed | hard | 혼합 풀, greedy 증류 | 포함 |

풀: mixed 6,400장/3,697장면(실내 50·주행 50), indoor 6,400장/5,867장면, outdoor 6,400장/509장면(주행은 장면 수가 509에서 포화 → N 축 상단은 같은 장면의 프레임 추가, 한계로 명시). 격자 8셀: 이미지 수 N ∈ {400, 1600, 6400} × 이미지당 픽셀 k ∈ {1, 4, 16}, N·k ≤ 25,600. 셀당 2 epoch, LoRA r16, 학생 입력 초점 750.
예상 시간(RTX PRO 4500 기준 0.46 s/step): 작업당 학습 ≈ 19 h + 평가 ≈ 9 h. H200 은 이보다 짧을 것으로 예상(실측 필요).

## 환경

- Python ≥ 3.10, `pip install -r requirements.txt` (torch 는 CUDA 에 맞는 빌드로; transformers 5.16.1, peft ≥ 0.20 고정)
- 모델: `Qwen/Qwen2.5-VL-3B-Instruct` (Apache-2.0, 7.5 GB, 자동 다운로드). 교사 `facebook/DepthLM` 은 이 저장소의 학습·평가에서는 **불필요** (라벨이 포함됨).
- 격자 작업은 HF_TOKEN 이 필요 없다(학생 모델 Apache-2.0). 라벨링 작업만 교사(gated) 다운로드에 읽기 전용 토큰이 필요하다.
- 기본 이미지: `Dockerfile` 참조 (pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime + requirements.txt).
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
