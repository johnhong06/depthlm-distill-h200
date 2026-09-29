# 풀 v5: 거의 같은 이미지를 순서 뒤로 (mixed, 2026-09-29)

- 규칙: dHash 해밍 ≤ 10 이고 32×32 회색조 상관 > 0.9 인 이미지가 이미 남긴 이미지 중에 있으면 뒤로 (`experiments/07_dedup_pool_order.py`)
- 뒤로 보낸 이미지 749장, 소스별 {'kitti_raw': 346, 'kitti2015': 193, 'kitti2012': 146, 'kitti_dsel': 64}
- 서로 다른 이미지 수 (앞 N 장 안): 전 {N400: 391, N1600: 1388, N6400: 5651} → 후 {N400: 400, N1600: 1600, N6400: 5651}
- 장면 라벨 수 (앞 N 장): 전 {N400: 400, N1600: 1296, N6400: 3684} → 후 {N400: 400, N1600: 1264, N6400: 3684}
- 도메인 비율 (앞 N 장, 전 = 후): {N400: {'driving': 200, 'indoor': 200}, N1600: {'driving': 800, 'indoor': 800}, N6400: {'driving': 3200, 'indoor': 3200}}
- 소스 구성 (앞 1600 장): 전 {'sunrgbd_kv2': 445, 'kitti2012': 299, 'kitti2015': 296, 'kitti_raw': 179, 'sunrgbd_rs': 132, 'sunrgbd_xtion': 101, 'sunrgbd_b3do': 63, 'nyuv2': 59, 'kitti_dsel': 26} → 후 {'sunrgbd_kv2': 445, 'kitti_raw': 300, 'kitti2012': 242, 'kitti2015': 207, 'sunrgbd_rs': 132, 'sunrgbd_xtion': 101, 'sunrgbd_b3do': 63, 'nyuv2': 59, 'kitti_dsel': 51}
- 필요한 라벨 44800 px 중 기존 라벨에 없는 2544 px → `bash run.sh label mixed` 가 이것만 라벨링한다

뒤로 보낸 이미지와 짝(처음 20개):

- `pool/kitti/raw/frames/2011_09_28_drive_0075/0000000040.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0057/0000000000.png`
- `pool/kitti/kitti2015/training/image_2/000011_10.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0009/0000000430.png`
- `pool/kitti/kitti2015/training/image_2/000092_11.png` ≈ `pool/kitti/kitti2015/training/image_2/000097_11.png`
- `pool/kitti/kitti2015/training/image_2/000095_10.png` ≈ `pool/kitti/kitti2015/training/image_2/000097_11.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0043/0000000110.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0039/0000000010.png`
- `pool/kitti/kitti2015/training/image_2/000093_11.png` ≈ `pool/kitti/kitti2015/training/image_2/000097_11.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0021/0000000090.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0016/0000000070.png`
- `pool/kitti/kitti2015/training/image_2/000052_10.png` ≈ `pool/kitti/kitti2015/training/image_2/000047_10.png`
- `pool/kitti/kitti2015/training/image_2/000090_10.png` ≈ `pool/kitti/kitti2015/training/image_2/000097_11.png`
- `pool/kitti/raw/frames/2011_09_26_drive_0032/0000000350.png` ≈ `pool/kitti/kitti2015/training/image_2/000079_11.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0082/0000000000.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0057/0000000000.png`
- `pool/kitti/raw/frames/2011_09_28_drive_0039/0000000320.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0039/0000000010.png`
- `pool/kitti/kitti2015/training/image_2/000051_10.png` ≈ `pool/kitti/kitti2015/training/image_2/000047_10.png`
- `pool/kitti/kitti2015/training/image_2/000094_11.png` ≈ `pool/kitti/kitti2015/training/image_2/000097_11.png`
- `pool/kitti/kitti2015/training/image_2/000091_10.png` ≈ `pool/kitti/kitti2015/training/image_2/000097_11.png`
- `pool/kitti/kitti2015/training/image_2/000113_10.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0057/0000000090.png`
- `pool/kitti/kitti2015/training/image_2/000048_11.png` ≈ `pool/kitti/kitti2015/training/image_2/000047_10.png`
- `pool/kitti/kitti2015/training/image_2/000096_10.png` ≈ `pool/kitti/kitti2015/training/image_2/000097_11.png`
- `pool/kitti/kitti2015/training/image_2/000114_11.png` ≈ `pool/kitti/raw/frames/2011_09_26_drive_0057/0000000090.png`
- `pool/kitti/kitti2015/training/image_2/000156_11.png` ≈ `pool/kitti/raw/frames/2011_09_28_drive_0016/0000000070.png`

참고: 혼합 풀은 도메인이 둘이라 도메인마다 따로 중복을 정리하고, 위치마다 원래 도메인의 다음 이미지로 채워 어느 N 에서든 실내·주행 비율이 원래와 같다. 실내 쪽 중복은 0장.
