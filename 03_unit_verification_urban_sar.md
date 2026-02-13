# Urban SAR 재수행 패키지 3/5: 단위기능 검증 (Unit Verification) (v1.2, 2026-02-11)

각 UF가 "독립적으로" 요구사항을 만족하는지 검증하는 테스트/실험 정의서.
각 테스트는 **입력/절차/산출물/수락 기준**을 포함한다.

------------------------------------------------------------------------

## 공통 산출물 템플릿

-   `results/UF-xx/<test_name>/`
    -   `config.yaml`
    -   `artifacts/` (npy/png/csv)
    -   `metrics.json`
    -   `visual_report/` (진단 플롯)
    -   `log.txt`

------------------------------------------------------------------------

# 1. UF-19 Facet/BRDF Scattering 검증

## UF19-T01: 평면 지면 BRDF sanity

-   입력: 평탄 DEM + uniform texture
-   절차: facet 생성 → normal 분포 확인 → rcs 분포 확인
-   수락: normal의 z성분 평균이 0.99 이상(평탄), rcs가 공간적으로
    균일(CV < 0.1, CV = std/mean)

## UF19-T02: 경사면 incidence 의존성

-   입력: 단일 경사면 DEM
-   절차: 룩각(또는 플랫폼 위치) 변화 스윕 (최소 5단계: 20°~60°)
-   수락: incidence 증가 시 평균 rcs가 단조 감소. 인접 단계간 감소폭 > 0.5 dB.

------------------------------------------------------------------------

# 2. UF-20 Shadow Masking 검증

## UF20-T01: 단일 벽 뒤 그림자 생성

-   입력: "벽(고도 step)"이 있는 DEM, 플랫폼 한쪽
-   절차: LOS 마스크 생성
-   수락: 벽 뒤 영역 vis_mask 평균 ≤ 0.1, 앞 영역 ≥ 0.9

## UF20-T02: 다중 벽 그림자 (신규)

-   입력: 두 개의 서로 다른 높이 벽이 있는 DEM
-   절차: LOS 마스크 생성, 각 벽 뒤 shadow 길이 비교
-   수락: 높은 벽 뒤 shadow가 낮은 벽 뒤보다 길어야 함 (최소 1.5배)

------------------------------------------------------------------------

# 3. UF-21 Layover Approx 검증

## UF21-T01: 급경사/수직 근사에서 layover flag

-   입력: 급경사면/박스 구조(단순 도시 블록)
-   절차: layover_flag 계산
-   수락: 급경사 영역에서 layover_flag 비율이 평탄 영역 대비 유의미하게
    큼(예: 5배)

------------------------------------------------------------------------

# 4. UF-22 좌표 통합 검증

## UF22-T01: round-trip 오차

-   입력: 임의 scatterer 좌표(DEM 상 랜덤 샘플, 최소 100개)
-   절차: CRS→ENU→CRS 왕복
-   수락: RMSE < 0.5 pixel(DEM spacing 기준) 또는 < 0.5 m(선택)

## UF22-T02: texture↔dem 정합 확인

-   입력: 동일 랜드마크 포인트 리스트(수동/자동)
-   수락: 랜드마크 대응 오차 < 1 pixel

------------------------------------------------------------------------

# 5. UF-24 Adaptive Threshold 검증

## UF24-T01: 목표 밀도 유지

-   입력: 도시 블록 장면(산란체 매우 많음)
-   절차: 목표 scatterer_budget 설정 후 필터
-   수락: 결과 scatterer 수가 목표의 ±10% 이내

## UF24-T02: 구조 보존

-   입력: edge-rich texture 장면
-   절차: 필터 전/후 edge 주변 산란체 비율 비교
-   수락: edge 주변 산란체 비율이 전체 대비 유지되거나 증가(≥ baseline)

------------------------------------------------------------------------

# 6. UF-25 Power Normalization 검증

## UF25-T01: range 감쇠 보정

-   입력: 거리 분포가 넓은 scatterers + 동일 rcs
-   절차: 보정 on/off 비교
-   수락: 거리별 평균 amplitude 편차가 30% 이상 감소

------------------------------------------------------------------------

# 7. UF-27 Streaming BP 검증

## UF27-T01: 타일링 on/off 일치성

-   입력: 점 타겟 장면, 작은 ROI
-   절차: (a) 전체 BP (b) 타일+overlap+stitch
-   수락: peak-normalized RMS diff < 1e-3 (정규화: peak amplitude로 나눔)

## UF27-T02: VRAM 상한 준수

-   입력: 중간 ROI(예: 1024²), K 증가
-   절차: vram_limit_gb 설정 후 실행
-   수락: 메모리 사용량이 상한 초과하지 않음(로그로 증명).
    측정: NumPy fallback 시 process RSS, GPU 시 CuPy mempool.used_bytes()

## UF27-T03: 선형 스케일링

-   입력: ROI 크기 512²→1024²→2048²
-   절차: 동일 K에서 런타임 측정
-   수락: 런타임이 면적에 대체로 선형(편차 ±20%)

------------------------------------------------------------------------

# 8. UF-26 Metrics 검증

## UF26-T01: ENL 계산 sanity

-   입력: homogeneous patch(균일 스펙클 가정)
-   수락: ENL이 multilook 수 증가에 따라 증가

## UF26-T02: edge-correlation

-   입력: DEM edge map과 SAR edge map
-   수락: UF-22 정합 후 correlation이 baseline 대비 증가

------------------------------------------------------------------------

# 9. UF-28 Raw Echo Generation 검증 (신규)

## UF28-T01: 점 타겟 에너지 보존

-   입력: 단일 점 타겟 (σ=1.0, scene center)
-   절차: raw echo 생성 → 총 에너지(|s|² 합) 계산
-   수락: 에너지가 σ²×K에 비례 (허용 오차 ±10%)

## UF28-T02: range compressed echo peak 위치

-   입력: 두 점 타겟 (서로 다른 range 위치)
-   절차: range compressed echo에서 peak 위치 측정
-   수락: peak 간 거리가 실제 range 차이와 일치 (오차 < 1 range bin)

------------------------------------------------------------------------

# 10. UF-29 Visual BP Verification 검증 (신규)

## UF29-T01: 점 타겟 진단 플롯 생성 검증

-   입력: 점 타겟 BP 결과 (SLC)
-   절차:
    1. visual verification 모듈 실행
    2. 진단 플롯 파일 존재 확인
    3. 에이전트가 생성된 intensity_db.png를 직접 열어 점 타겟 PSF 확인
    4. range_cut.png / azimuth_cut.png에서 mainlobe/sidelobe 구조 시각 확인
-   수락:
    -   모든 플롯 파일 생성됨 (intensity_db.png, phase.png, histogram.png,
        range_cut.png, azimuth_cut.png, irf_2d_contour.png)
    -   visual_metrics.json의 dynamic_range_db ≥ 30
    -   visual_metrics.json의 psf_aspect_ratio ≤ 1.5
    -   **에이전트 시각 판정: PSF가 중심에 위치하고 대칭적 sidelobe 패턴 확인**

## UF29-T02: 도시 장면 진단 플롯 생성 검증

-   입력: 도시 블록(box) DEM + shadow/layover가 있는 BP 결과
-   절차:
    1. visual verification 모듈 실행 (scene_type="urban")
    2. shadow_overlay.png, layover_overlay.png 생성 확인
    3. 에이전트가 shadow_overlay.png를 직접 열어 shadow 영역이
       벽 뒤에 올바르게 위치하는지 시각 확인
-   수락:
    -   shadow_overlay.png에서 shadow가 벽 반대편에 위치 (시각 확인)
    -   intensity_db.png에서 도시 구조가 시각적으로 구분됨
    -   **에이전트 시각 판정: shadow 영역이 기하학적으로 일관되고,
        건물 구조가 intensity 이미지에서 구분 가능**

------------------------------------------------------------------------

# 11. UF-30 Texture Loader 검증 (v1.2 신규)

## UF30-T01: TIF→ENU 좌표 변환 정확성

-   입력: 알려진 특징이 있는 TIF (예: 왼쪽 아래에 밝은 점)
-   절차:
    1. load_texture_tif() 실행
    2. 결과 texture_gray에서 해당 특징이 ENU 좌표 (row 0=south, col 0=west)에 올바르게 위치하는지 확인
    3. TIF의 bottom-left 특징이 texture_gray[0, 0] 근처에 있어야 함
-   수락: 수직 플립 후 좌표 대응 정확 (bottom-left 보존)

## UF30-T02: 텍스처 기반 RCS 분포

-   입력: 도시 항공사진 TIF (건물+도로+녹지 혼합)
-   절차:
    1. texture_to_rcs() 실행
    2. RCS 값 범위, 분포 확인
    3. 엣지 부스팅이 건물 경계에서 RCS를 높이는지 확인
-   수락:
    -   rcs_min ≤ RCS ≤ rcs_max × edge_boost
    -   건물 경계 픽셀의 평균 RCS > 전체 평균 RCS

## UF30-T03: Pseudo-DEM 건물 감지

-   입력: 도시 항공사진 TIF
-   절차:
    1. texture_to_flat_dem() 실행
    2. 밝은 영역(건물)이 building_height로, 어두운 영역이 base_height로 할당되는지 확인
-   수락:
    -   elevated 픽셀 비율이 10%~50% 범위 (도시 지역 합리적 범위)
    -   형태학적 정리로 1-pixel 노이즈 제거됨

------------------------------------------------------------------------

# 12. UF-31 Speckle Noise 검증 (v1.2 신규)

## UF31-T01: 스페클 통계 검증

-   입력: 균일 배경 SAR 이미지 (flat DEM, 균일 RCS)
-   절차:
    1. speckle noise 추가 (ENL=1)
    2. intensity 분포 측정
-   수락:
    -   intensity 분포가 exponential distribution에 근사 (K-S test p>0.01)
    -   intensity CV ≈ 1.0 (single-look speckle 이론값)

## UF31-T02: ENL 조절 검증

-   입력: 동일 SAR 이미지, ENL=1 vs ENL=4
-   절차:
    1. 각 ENL로 speckle 추가
    2. 균일 영역에서 measured ENL 비교
-   수락: ENL=4 적용 시 measured ENL ≥ 3.0 (이론적 4.0에 근사)

------------------------------------------------------------------------

# 13. UF-32 Performance Profiler 검증 (v1.2 신규)

## UF32-T01: 스테이지별 시간 기록

-   입력: 임의 파이프라인 실행
-   절차:
    1. E2E 파이프라인 실행
    2. metrics.json에 stage_timings 포함 확인
-   수락: scene_time_s, raw_time_s, bp_time_s, total_time_s 모두 기록됨
