# Urban SAR 재수행 패키지 4/5: 통합(Integration) 설계 (v1.2, 2026-02-11)

본 문서는 UF들을 `pipeline/e2e.py` 기준으로 통합하는 설계를 정의한다.

------------------------------------------------------------------------

## 1. 통합 원칙

-   scene/physics(UF-19\~25)는 raw 생성 이전(Pre-stage)에 위치
-   좌표 통합(UF-22)은 scene 생성 직후, BP 직전에 적용
-   BP 성능(UF-27)은 UF-09(method="bp") 내부 구현 옵션
-   metrics(UF-26)은 UF-17을 보강하며 결과/CI 회귀에 사용
-   UF-28 (raw echo)은 scene→imaging 사이 필수 단계
-   UF-29 (visual verification)은 이미징 후 최종 단계
-   **Config 검증은 파이프라인 진입 전에 수행 (fail-fast)**
-   **각 Stage 실패 시 즉시 중단 + 에러 로그 저장 (no silent failure)**

------------------------------------------------------------------------

## 2. 통합 파이프라인(권장 순서)

### Stage-0 Config Validation (신규)

-   config.yaml 로드 및 스키마 검증
-   필수 키 존재 확인, 타입 검사
-   output 디렉토리 생성

### Stage-1 Scene Pre-stage

1)  합성 DEM 생성 (또는 GeoTIFF 로드)
2)  UF-22 unified ENU 좌표 변환
3)  UF-19 facet/BRDF scattering
4)  UF-20 shadow mask
5)  UF-21 layover approx
6)  UF-24 adaptive threshold (옵션)
7)  UF-25 beam/power normalization (옵션)

### Stage-2 Platform & Raw Generation

-   합성 플랫폼 궤적 생성 (SR-04b)
-   UF-28 raw echo generation

### Stage-3 Imaging

-   BP 체인:
    -   UF-09 method="bp" (내부에 UF-27 streaming)
-   (옵션) UF-10 autofocus

### Stage-4 Metrics/Product

-   UF-17 + UF-26 → quality metrics
-   UF-18 product format (SLC npy + intensity png)

### Stage-5 Visual Verification (신규)

-   UF-29 visual BP verification
-   진단 플롯 생성 → visual_report/ 저장
-   시각적 판정 지표 → visual_metrics.json

------------------------------------------------------------------------

## 3. 통합 설정 스키마(config) 핵심

-   `experiment.id`, `experiment.description`
-   `seed`, `output.run_id`, `output.root_dir`
-   `scene.build_in_e2e`: bool
-   `scene.dem_path` 또는 `scene.synthetic.type` (flat/slope/box)
-   `scene.texture_path`: TIF 파일 경로 (실제 데이터 입력, v1.2 신규)
-   `scene.grid_spacing_m`: 목표 그리드 간격 [m] (실제 데이터 시 다운샘플링 제어)
-   `scene.rcs_min`, `scene.rcs_max`: 텍스처→RCS 매핑 범위 (v1.2)
-   `scene.building_detection.threshold`, `.building_height_m`, `.base_height_m` (v1.2)
-   `scene.synthetic.size`, `scene.synthetic.spacing_m`
-   `scene.roi_enu` (xmin,xmax,ymin,ymax) in meters
-   `imaging.method`: bp (기본)
-   `scatter.model`: centroid/facet_brdf
-   `scatter.enable_shadow`, `scatter.enable_layover`
-   `scatter.multibounce.depth`: 0|1|2
-   `scatter.brdf.alpha`, `scatter.brdf.beta`, `scatter.brdf.p`
-   `bp.enable_streaming`, `bp.vram_limit_gb`
-   `bp.tile_size: [Nx, Ny]`, `bp.k_batch`
-   `bp.overlap`, `bp.stitch_method`
-   `radar.fc_hz`, `radar.bandwidth_hz`, `radar.prf_hz`, `radar.tp_s`
-   `platform.altitude_m`, `platform.velocity_m_s`, `platform.look_angle_deg`
-   `visual.enable`: bool, `visual.scene_type`: point_target/urban
-   `speckle.enable`: bool (v1.2), `speckle.enl`: float (기본 1.0)

------------------------------------------------------------------------

## 4. 에러 처리 전략 (신규)

-   Config 검증 실패: 즉시 종료 + 에러 메시지 (어떤 키가 문제인지 명시)
-   Stage 실패: 해당 stage에서 중단, 이전 stage 산출물은 보존
-   메모리 초과: 자동 tile_size 축소 재시도 1회, 그래도 실패 시 중단

------------------------------------------------------------------------

## 5. 로깅 전략 (신규)

-   Python logging 모듈 사용
-   레벨: INFO (기본), DEBUG (config로 전환 가능)
-   포맷: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
-   파일: `results/<run_id>/log.txt` + console 출력

------------------------------------------------------------------------

## 6. 통합 산출물 구조(재현성)

-   `results/<run_id>/`
    -   `config.yaml`
    -   `meta.json`(seed/timestamp)
    -   `slc.npy`, `intensity.png`
    -   `metrics.json` (stage_timings 포함, v1.2)
    -   `visual_report/` (진단 플롯 모음)
        -   `texture_comparison.png` (실제 데이터 시, v1.2 신규)
    -   `visual_metrics.json`
    -   `log.txt`

------------------------------------------------------------------------

## 7. Stage-1 실제 데이터 경로 (v1.2 신규)

`scene.texture_path`가 설정되고 파일이 존재하면:

1.  UF-30 load_texture_tif() → ENU 정렬 grayscale + 좌표축
2.  UF-30 texture_to_flat_dem() → pseudo-DEM
3.  UF-30 texture_to_rcs() → RCS map
4.  UF-22 create_bp_grid_from_dem() → BP grid
5.  UF-19 facet_brdf_scatterers() → 산란체 (texture RCS 곱셈 적용)
6.  UF-20/21/25 → shadow/layover/power norm (기존과 동일)

핵심 좌표 정합:
-   TIF row 0 = top (north) → ENU row 0 = south: 수직 플립 필수
-   ENU 축은 scene 중심 기준 ±(extent/2)로 배치
-   BP grid, scatterer 좌표, 텍스처 좌표가 모두 동일 ENU 공간 사용

------------------------------------------------------------------------

## 8. Stage-3.5 스페클 노이즈 (v1.2 신규)

BP 복원 직후, metrics 계산 전에 선택적으로 UF-31 적용:

-   `speckle.enable: true`이면 UF-31 실행
-   원본 SLC는 `slc_clean.npy`로 보존, speckled 버전은 `slc.npy`로 저장
-   metrics는 speckled 이미지 기준
