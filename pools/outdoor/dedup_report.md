# 풀 v5: 거의 같은 이미지를 순서 뒤로 (outdoor, 2026-09-29)

- 규칙: dHash 해밍 ≤ 10 이고 32×32 회색조 상관 > 0.9 인 이미지가 이미 남긴 이미지 중에 있으면 뒤로 (`experiments/07_dedup_pool_order.py`)
- 뒤로 보낸 이미지 1365장, 소스별 {'kitti_raw': 893, 'kitti2015': 195, 'kitti2012': 146, 'kitti_dsel': 131}
- 서로 다른 이미지 수 (앞 N 장 안): 전 {N400: 372, N1600: 1130, N6400: 5035} → 후 {N400: 400, N1600: 1600, N6400: 5035}
- 장면 라벨 수 (앞 N 장): 전 {N400: 400, N1600: 509, N6400: 509} → 후 {N400: 400, N1600: 476, N6400: 509}
- 소스 구성 (앞 1600 장): 전 {'kitti_raw': 718, 'kitti2015': 400, 'kitti2012': 388, 'kitti_dsel': 94} → 후 {'kitti_raw': 1017, 'kitti2012': 242, 'kitti2015': 205, 'kitti_dsel': 136}
- 필요한 라벨 44800 px 중 기존 라벨에 없는 5640 px → `bash run.sh label outdoor` 가 이것만 라벨링한다

뒤로 보낸 이미지와 짝(처음 20개):

- `pool/kitti/kitti2015/training/image_2/000048_10.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0017/0000000045.png`
- `pool/kitti/kitti2015/training/image_2/000096_11.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0051/0000000200.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0021/0000000080.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0016/0000000040.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0082/0000000030.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0057/0000000010.png`
- `pool/kitti/kitti2015/training/image_2/000155_10.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0002/0000000225.png`
- `pool/kitti/kitti2015/training/image_2/000047_10.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0017/0000000045.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0075/0000000005.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0057/0000000010.png`
- `pool/kitti/kitti2015/training/image_2/000036_10.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0015/0000000210.png`
- `pool/kitti/raw/frames/2011_09_26_drive_0059/0000000310.png` ≈ `pool/kitti/kitti2015/training/image_2/000126_11.png`
- `pool/kitti/kitti2015/training/image_2/000092_10.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0051/0000000200.png`
- `pool/kitti/kitti2015/training/image_2/000089_11.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0051/0000000200.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0039/0000000330.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0039/0000000045.png`
- `pool/kitti/kitti2015/training/image_2/000016_11.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0011/0000000230.png`
- `pool/kitti/kitti2015/training/image_2/000053_11.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0017/0000000045.png`
- `pool/distill_pool/rgb/kitti_000179_10.png` ≈ `pool/distill_pool/rgb/kitti_000170_10.png`
- `pool/kitti/kitti2015/training/image_2/000095_10.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0051/0000000200.png`
- `pool/kitti/kitti2015/training/image_2/000113_10.png` ≈ `pool/kitti/kitti2015/training/image_2/000115_10.png`
- `pool/distill_pool/rgb/kitti_000077_11.png` ≈ `pool/distill_pool/rgb/kitti_000017_11.png`
- `pool/kitti/kitti2015/training/image_2/000156_10.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0016/0000000040.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0002/0000000365.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0002/0000000225.png`

참고: DepthLM 은 비슷한 영상 프레임을 일부만 썼다고만 밝히고 규칙은 적지 않았다. 이 규칙(상관 > 0.9)은 그 단계를 명시한 버전이며, 경계 구간(0.90–0.93) 쌍도 눈으로 보면 같은 장면에서 차가 몇 m 움직인 사진이다. 0.97 로 올리면 0.1 초 간격 _10/_11 쌍의 다수(0.90–0.97)를 놓친다.
