# depthlm-distill-h200 (한국어)

영문 원본: [../README.md](../README.md). 이 페이지는 사업단 이슈 작성용 요약이다.

DepthLM(12B) → Qwen2.5-VL-3B 증류 실험을 비대화형 컨테이너(사업단 H200, GitHub Issue → Jenkins)에서 돌리기 위한 저장소.
혼합 풀의 교사 라벨은 저장소에 포함(`pools/mixed/teacher_labels.parquet`)되어 있고 실내·실외 풀 라벨은 파드에서 만든다.

## 이슈(요청서) 작성 값 — Aerodrone-H200 저장소의 "컨테이너 생성 및 코드 실행 요청" 템플릿

| 항목 | 값 |
|---|---|
| 사용자 ID | `johnhong06` |
| GitHub 링크 | `https://github.com/johnhong06/depthlm-distill-h200.git` |
| 사용 이미지 | `kau/pytorch-master` (CUDA 13, torch 2.11) |
| 사용 언어 | `Python` (예시 이슈와 동일. bash 스크립트를 실행해도 Python 으로 적음) |
| 추가 필요 모듈 (칸이 있을 때만) | `transformers==5.16.1 peft==0.20.0 pyarrow pyyaml tabulate` — 칸이 없어도 `run.sh` 가 시작할 때 스스로 설치함 |
| 실행 명령어 | 아래 표 |
| GPU 할당량 | 아래 표 (`1` = 18 GB 슬라이스, `7` = GPU 한 장 통째) |

| 이슈 | 실행 명령어 | GPU | 비고 |
|---|---|---|---|
| ① 스모크 | `bash run.sh smoke` | 1 | 모델 다운로드 → 30 스텝 → 3 px 평가. 10 분. 데이터 불필요 |
| ② 전체 체인 | `bash run.sh all` (토큰은 `/app/data/hf_token.txt` 또는 인자 `hf_xxx`) | **7** | 혼합 격자 soft·hard → 실내·실외 라벨링(교사 30 GB × 4 병렬) → 실내·실외 격자 4개. 한 단계가 실패해도 다음으로 넘어감. 토큰은 라벨링에만 쓰이며 읽기 전용, 끝나면 폐기 |

나눠서 내고 싶으면 단계별 명령도 그대로 쓸 수 있다: `bash run.sh grid <mixed|indoor|outdoor> <soft|hard>`, `bash run.sh label <indoor|outdoor> hf_xxx`. 격자는 저장소의 `pools/<pool>/teacher_labels.parquet` 가 없으면 파드에서 만든 `/app/output/labels/<pool>/teacher_labels.parquet` 를 쓴다. 이미 끝난 셀·평가·라벨은 건너뛰므로 같은 명령을 다시 내면 이어서 돈다.

- 신청 창 안에서 이슈를 순서대로 낸다. 컨테이너는 끝나면 삭제되고 `/app/output/` 만 남는다(관리자에게 파일 요청). 리포트는 65,000자까지만 오므로 표준 출력은 요약, 전체 로그는 `/app/output/` 에 쓴다. 격자마다 `results_<cond>_<pool>.zip`(체크포인트·평가·표·그림·로그) 이 만들어지고 라벨은 `labels/<pool>/teacher_labels.parquet` 에 남는다. 관리자에게 zip 6개와 라벨 2개를 요청하면 된다.
- 데이터는 사업단이 `/app/data/` 에 넣어 준 tar 분할본을 첫 실행 때 `/app/output/data/` 에 푼다(미리 풀어 두면 그대로 씀). 모델 가중치는 실행 중 Hugging Face 에서 받는다(학생 Apache-2.0, 토큰 불필요; 교사는 gated → 토큰을 `/app/data/hf_token.txt` 파일로 관리자에게 전달하거나 명령 인자로 넘김).
- GPU `7` 이면 스크립트가 자동으로 학습 8 병렬·라벨링 4 병렬로 돈다(`NPROC`, `NPROC_LABEL` 로 변경 가능). `1` 이면 순차.

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
- 기본 이미지: 사업단 `kau/pytorch-master`. 추가 모듈은 requirements.txt 의 한 줄.
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

MIT(`LICENSE`)는 이 저장소를 위해 작성한 코드에만 적용된다. 교사 의사 라벨(`pools/*/teacher_labels.parquet`)과 그것으로 학습한 어댑터는 DepthLM 모델의 출력물이라 FAIR Noncommercial Research License(사본 `third_party/DepthLM_Official/MODEL_LICENSE`)에 따라 비상업 연구용으로만 쓸 수 있고 논문에 DepthLM 사용을 밝혀야 한다. 벤더링한 DepthLM 코드(`third_party/`)는 CC BY-NC 4.0. `ref/`와 `smoke/data/`의 픽셀 깊이 값·소수 이미지는 각 데이터셋(iBims-1, NYUv2, ETH3D, SUN RGB-D, KITTI)의 연구용 조건을 따르며 데이터셋 전체는 재배포하지 않는다. 상세는 `NOTICE`.
