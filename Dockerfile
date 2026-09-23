# 사업단 H200 파드용. 기본 이미지: PyTorch 2.8 + CUDA 12.8 (H200/Hopper 지원). 실행 명령은 run.sh 참조.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
WORKDIR /app/repo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# 데이터: /app/data (사업단 마운트), 결과: /app/output (보존). 예) MODE=grid POOL=mixed COND=soft bash run.sh
CMD ["bash", "run.sh"]
