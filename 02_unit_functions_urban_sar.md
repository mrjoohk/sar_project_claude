# Urban SAR 재수행 패키지 2/5: 단위기능 (Unit Functions) (v1.2, 2026-02-11)

본 문서는 요구사항을 **UF(단위기능)**로 분해하고, 각 UF의
입력/출력/제한사항/구현 노트를 명시한다. 기존 체인(UF-01\~UF-18)에
신규(UF-19\~UF-32)를 결합한다.

------------------------------------------------------------------------

## 공통 데이터 계약(Data Contracts)

### Meta

-   `meta.platform`: t_k, p_k(x,y,z), v_k, attitude_k
-   `meta.radar`: fc, B, PRF, Tp, (옵션) f_n
-   `meta.geo`: crs, enu_origin, transforms

### Scatterers (SoA 권장)

-   필수: `sx, sy, sz (float32)`, `s_rcs (float32)`
-   옵션: `snx, sny, snz`, `s_area`, `mat_id`, `vis_mask`,
    `layover_flag`

### BP Grid

-   `bp_grid_xyz` 또는 SoA(px,py,pz)

### Raw

-   time-domain: `raw[K,Nr] complex64`
-   freq-domain: `raw[K,Nf] complex64`

------------------------------------------------------------------------

# A. 기존 UF-01 \~ UF-18 (요약)

## UF-01 ingest/validate

-   입력: raw + header/meta
-   출력: validated raw, meta
-   제한: K 축은 meta.t_k와 매핑되어야 함

## UF-02 metadata_align

-   입력: meta.platform
-   출력: aligned state vectors/attitude

## UF-03 timing_prf_validate

-   입력: PRF, t_k
-   출력: timing corrected grid

## UF-04 range_resample

-   입력: raw(time) or range axis
-   출력: resampled raw / axis
-   제한: BP에서 raw-domain 보간을 하는 경우 옵션

## UF-05\~UF-10 표준 체인

-   UF-05 range compress
-   UF-06/06b doppler centroid est/corr
-   UF-07 motion compensate
-   UF-08 RCMC
-   UF-09 azimuth compress (FFT) / 또는 method="bp"
-   UF-10 autofocus

## UF-11\~UF-13 보정/멀티룩/스펙클

-   UF-11 radiometric calibration(상대 보정)
-   UF-12 multilook
-   UF-13 speckle filter

## UF-14\~UF-16 geocode/orthorectify/mosaic

-   UF-14 geocode
-   UF-15 orthorectify
-   UF-16 tile mosaic

## UF-17\~UF-18 metrics/product

-   UF-17 quality metrics(SNR/PSLR/ISLR)
-   UF-18 product format

------------------------------------------------------------------------

# B. 신규 UF-19 \~ UF-29 (정의)

## UF-19 Facet/BRDF Scattering

### 목적

DEM centroid 모델을 facet/patch 기반 + BRDF(Lambert+Phong)로 확장

### 입력

-   DEM mesh triangles(또는 grid→triangulation)
-   (옵션) texture-aligned reflectance map/edge map
-   (옵션) material priors (ρ_d, ρ_s, p)

### 출력

-   scatterers(SoA): `sx,sy,sz,s_rcs` (+ `snx,sny,snz,s_area` 옵션)

### 제한사항

-   full EM 아님(근사)
-   facet 해상도는 DEM grid에 의해 제한

### 구현 노트

-   triangle centroid를 대표 산란점으로 쓰되, `normal/area`를 유지하여
    BRDF 가중을 적용
-   BRDF 기본 계수: α=0.7(diffuse), β=0.3(specular), p=10(Phong exponent)
-   수식: σ = A × [α × cos(θ_i) + β × |r_hat · s_hat|^p]
    여기서 A=facet area, θ_i=입사각, r_hat=반사 방향, s_hat=specular 방향

------------------------------------------------------------------------

## UF-20 Shadow Masking (LOS 근사)

### 입력

-   platform geometry (대표 시점 또는 t_k)
-   DEM/mesh, scatterers

### 출력

-   `vis_mask` 또는 `s_rcs *= vis_mask`

### 제한사항

-   ray-march는 비용 큼 → height test 우선, 필요 시 sparse ray

### 구현 노트

-   기본 알고리즘: height-profile test
    1. 플랫폼→산란체 LOS 벡터를 DEM 그리드 위에 투영
    2. LOS 경로상 DEM 고도가 산란체→플랫폼 직선보다 높으면 shadow
-   vis_mask는 [0, 1] float (soft shadow 가능, 기본은 binary 0/1)

------------------------------------------------------------------------

## UF-21 Layover Approx

### 입력

-   facet geometry(normal, slope), incidence angle 범위

### 출력

-   `layover_flag` 또는 `layover_weight`

### 제한사항

-   정밀 layover 시뮬레이션이 아니라 "누적 허용/가중" 수준

### 구현 노트

-   layover 조건: local slope angle > incidence angle (레이더 쪽으로 기울어진 면)
-   layover_flag = 1 if slope_angle > incidence_angle, else 0
-   layover_weight = max(0, sin(slope_angle - incidence_angle)) for soft version

------------------------------------------------------------------------

## UF-22 Unified Scene Coordinate System (CRS→ENU 통일)

### 입력

-   DEM CRS + enu_origin
-   scatterers, bp_grid_xyz, rcs_map(옵션)

### 출력

-   통일 좌표계 데이터(ENU), 변환 매트릭스/함수

### 제한사항

-   절대 지오 정합을 주장하려면 추가 보정 필요

### 구현 노트

-   pyproj 기반 CRS→ECEF→ENU 변환
-   enu_origin = DEM centroid 또는 config 지정
-   round-trip 오차 < 0.5 pixel 보장을 위해 float64로 변환 연산 수행

------------------------------------------------------------------------

## UF-23 RD Geocode (옵션)

### 입력

-   SAR image + meta + DEM

### 출력

-   geocoded product + geo error stats

### 제한사항

-   ROI 유지와 절대 정합은 trade-off

------------------------------------------------------------------------

## UF-24 Adaptive Scatterer Threshold

### 입력

-   scatterer rcs 분포, 목표 밀도/예산

### 출력

-   필터링된 scatterers + threshold report

### 제한사항

-   과도한 필터링은 구조 손실

### 구현 노트

-   percentile 기반 threshold: target_count 달성을 위해
    rcs 분포에서 하위 percentile 제거
-   edge 주변 산란체 보존 가중치 옵션

------------------------------------------------------------------------

## UF-25 Beam/Power Normalization

### 입력

-   range/angle, beam gate 설정, platform geometry

### 출력

-   per-pulse gain 또는 `s_rcs` 스케일 보정 파라미터

### 제한사항

-   절대 방사 보정이 아니라 붕괴 방지/상대 정규화 목적

### 구현 노트

-   거리 감쇠 보정: gain = (R / R_ref)^n, n=2 (기본, 양방향 전파)
-   R_ref = scene center 거리
-   beam gate: 유효 거리 범위 [R_min, R_max] 밖 산란체 제거

------------------------------------------------------------------------

## UF-26 Robust Quality Metrics (UF-17 보강)

### 입력

-   SAR image, (옵션) DEM edge map, ROI mask

### 출력

-   metrics.json: ENL, CCR, edge-corr 등

### 제한사항

-   DEM↔SAR 비교는 정합(UF-22)이 선행되어야 의미 있음

------------------------------------------------------------------------

## UF-27 Streaming BP (UF-09 bp 내부 또는 UF-09a)

### 입력

-   raw, meta(platform), bp_grid_xyz
-   config: `M_tile, K_batch, vram_limit_gb`

### 출력

-   tile-wise image + stitched image

### 제한사항

-   O(MK) 근본 복잡도는 변하지 않음

### 구현 노트

-   NumPy 기반 (CuPy RawKernel은 GPU 있을 때 옵션)
-   타일링: ROI를 M_tile×M_tile로 분할, overlap 포함
-   stitch_method: "linear_blend" (기본) — overlap 영역에서 선형 가중 합산
-   K_batch: 한 번에 처리하는 pulse 수 (메모리 제어)

------------------------------------------------------------------------

## UF-28 Raw Echo Generation (신규)

### 목적

scatterer list + platform trajectory → raw IQ 데이터 생성 (SR-04 충족)

### 입력

-   scatterers(SoA): sx, sy, sz, s_rcs
-   meta.platform: t_k, p_k(x,y,z)
-   meta.radar: fc, B, Tp, PRF

### 출력

-   `raw[K, Nr] complex64` (time-domain)

### 제한사항

-   chirp convolution 근사 (matched filter 관점에서 range compressed echo 직접 생성 옵션)

### 구현 노트

-   수식: s(k, t) = Σ_i σ_i × rect((t - 2R_i(k)/c) / Tp) × exp(-j4πfc R_i(k)/c)
-   range compressed 버전: s_rc(k, n) = Σ_i σ_i × sinc(B(τ_n - 2R_i(k)/c)) × exp(-j4πfc R_i(k)/c)
-   실용상 range compressed echo를 직접 생성하여 BP에 입력하는 것을 기본으로 함
-   **최적화 (v1.2)**: 희소 sinc 누적 — 각 산란체는 ±sinc_halfwidth 범위 빈에만 기여.
    Dense [N, Nr] 행렬 대신 np.add.at 기반 scatter-add로 O(K×N×W) 복잡도 (W=2×halfwidth+1≪Nr).
    124K 산란체, 256 pulse 기준 ~40초 (dense 방식 대비 ~15× 개선).

------------------------------------------------------------------------

## UF-29 Visual BP Verification (신규)

### 목적

BP 복원 결과 이미지를 자동으로 분석하고 진단 플롯을 생성하여
에이전트(사람 또는 자동)가 시각적으로 품질을 판정할 수 있게 한다. (SR-07 충족)

### 입력

-   SLC image (complex)
-   (옵션) vis_mask, layover_flag
-   (옵션) 점 타겟 위치 리스트
-   config: scene type (point_target / urban / custom)

### 출력

-   `visual_report/` 디렉토리:
    -   `intensity_db.png`: dB 스케일 intensity 이미지
    -   `phase.png`: phase 이미지
    -   `histogram.png`: intensity histogram (dynamic range 표시)
    -   `range_cut.png` / `azimuth_cut.png`: 1D cut (점 타겟 시)
    -   `irf_2d_contour.png`: IRF 2D contour (점 타겟 시)
    -   `shadow_overlay.png`: shadow 영역 오버레이 (도시 장면 시)
    -   `layover_overlay.png`: layover 영역 오버레이 (도시 장면 시)
-   `visual_metrics.json`: dynamic range, PSF aspect ratio, shadow contrast 등

### 제한사항

-   matplotlib 기반, 대형 이미지는 downsample 후 플롯

### 구현 노트

-   점 타겟 모드: peak 찾기 → ±N pixel crop → 4x zero-pad → contour + 1D cut
-   도시 모드: intensity(dB) + vis_mask overlay + layover overlay
-   **실제 데이터 모드**: 텍스처-SAR 비교 오버레이 (side-by-side + M/G overlay)
-   자동 판정 기준: AC-07, AC-08 참조

------------------------------------------------------------------------

## UF-30 Texture Loader (v1.2 신규)

### 목적

실제 도심 항공사진/위성영상 TIF를 SAR 시뮬레이션 입력으로 변환한다. (SR-08 충족)

### 입력

-   TIF 파일 경로 (RGB 또는 grayscale, CRS 유무 무관)
-   목표 grid spacing [m]
-   건물 감지 파라미터 (threshold, height)

### 출력

-   `x_axis[Nx]`, `y_axis[Ny]`: ENU 좌표축 [m]
-   `texture_gray[Ny, Nx]`: ENU 정렬된 grayscale [0..1] (row 0=south)
-   `rcs_map[Ny, Nx]`: 텍스처 기반 RCS 값
-   `dem[Ny, Nx]`: pseudo-DEM 높이맵 [m]

### 제한사항

-   pseudo-DEM은 건물 높이를 추정할 뿐이며, 실제 LiDAR DEM을 대체할 수 없음
-   CRS 없는 TIF는 pixel spacing 추정에 의존 (기본: 0.25m, 한국 정사영상 표준)

### 구현 노트

-   좌표 정합 핵심: `gray = gray[::-1, :]` (TIF row 0=north → ENU row 0=south)
-   RGB→grayscale: luminance 가중 (0.299R + 0.587G + 0.114B)
-   downsampling: block average (target_spacing / pixel_spacing 배율)
-   texture_to_rcs(): 밝기→RCS 선형 매핑 + Sobel 엣지 부스팅 (edge_boost 파라미터)
-   texture_to_flat_dem(): 밝기 임계값 + morphological opening으로 건물 감지

------------------------------------------------------------------------

## UF-31 Speckle Noise (v1.2 신규)

### 목적

BP 복원 영상에 현실적인 SAR 스페클 노이즈를 추가한다. (SR-09 충족)

### 입력

-   SLC image (complex)
-   config: ENL target (1=single look, >1=multi-look equivalent)
-   seed (재현성)

### 출력

-   speckled SLC image (complex)

### 구현 노트

-   single-look: I_speckle = I_clean × Gamma(1, 1) — exponential multiplicative
-   complex domain: amplitude × sqrt(exponential), phase += uniform(0, 2π)
-   multi-look (ENL=L): 1/L × sum of L independent single-look realizations
-   또는 Gamma(L, 1/L) multiplicative factor

------------------------------------------------------------------------

## UF-32 Performance Profiler (v1.2 신규)

### 목적

파이프라인 각 스테이지 실행 시간을 자동 측정하고 기록한다. (SR-10 충족)

### 입력

-   파이프라인 실행 컨텍스트

### 출력

-   `stage_timings` dict in metrics.json
-   총 실행 시간, scatterer/pixel 처리율 [Mpixel-pulses/s]

### 구현 노트

-   time.time() 기반 per-stage 측정 (이미 e2e.py에 구현, metrics.json에 통합 필요)
