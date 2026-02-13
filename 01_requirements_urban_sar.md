# Urban SAR 재수행 패키지 1/5: 요구사항 (Requirements) (v1.2, 2026-02-11)

본 문서는 기존 md
산출물(roadmap/수학모델/쿠다설계/복잡도/스펙/검증/통합설계)을 기반으로,
프로젝트를 **요구사항 → 단위기능 → 단위기능 검증 → 통합 → 통합
검증**으로 재구성하기 위한 1단계 문서다.

------------------------------------------------------------------------

## 1. 목표(Goal)

-   도심 DEM/텍스처 기반 장면에서 **구조적으로 해석 가능한 SAR 영상**을
    생성한다.
-   도시 특유의 현상(코너 반사/그림자/layover)을 **근사 물리 모델**로
    재현한다.
-   대규모 ROI에서도 **타일링+GPU 스트리밍 BP**로 연산/메모리 제약 내
    실행한다.
-   결과는 제품(이미지/메타/지표)과 로그로 **재현 가능**해야 한다.

------------------------------------------------------------------------

## 2. 범위(Scope)

### In-scope

-   입력: GeoTIFF DEM + Texture(정합 가능), 또는 합성 DEM(flat/slope/box),
    또는 **실제 도심 항공사진 TIF(텍스처 기반 pseudo-DEM + RCS 생성)**
-   장면 구성: centroid 또는 facet/BRDF 산란체 생성(1-bounce 기본,
    2-bounce 옵션)
-   플랫폼 궤적: 합성 직선/원형 궤적 생성 (stripmap/spotlight 모드)
-   Raw 데이터 생성: time-domain IQ (기본) 또는 freq-domain (step-frequency, 옵션)
-   **스페클 노이즈: 현실적 SAR 영상 모사를 위한 multiplicative speckle 추가**
-   이미징: BP(기본) + 표준 체인(옵션) + (옵션) hybrid
-   좌표/정합: local ENU 통합, (옵션) geocode/orthorectify
-   품질 지표: SNR/PSLR/ISLR + (확장) ENL/CCR/edge-correlation
-   검증: 통제 장면 + 도시 블록 장면, 자동 리포트
-   **시각적 검증: BP 복원 이미지의 자동 진단 플롯 생성 및 에이전트 기반 품질 판정**

### Out-of-scope (v1)

-   full-wave EM(MoM/FDTD), 재질 DB 기반 정밀 산란
-   고정밀 센서 캘리브레이션(절대 방사 보정)
-   대규모 멀티 GPU 분산(추후)

------------------------------------------------------------------------

## 3. 시스템 요구사항(System Requirements)

### SR-01 입력 호환성

-   SR-01.1 GeoTIFF DEM을 읽고 CRS 정보를 획득할 수 있어야 한다.
-   SR-01.2 Texture를 DEM과 동일 좌표계/영역으로 정합(resample)할 수
    있어야 한다.
-   SR-01.3 합성 DEM(flat/slope/box 등)을 프로그래밍 방식으로 생성할 수
    있어야 한다.

### SR-02 좌표/정합

-   SR-02.1 DEM/Texture/RCS/bp_grid/scatterers는 **단일 local ENU
    좌표계**로 통일 가능해야 한다.
-   SR-02.2 scatterer 좌표를 bp_grid로 투영/역변환할 때, ROI 내부에서
    **오차 ≤ 1 pixel**을 목표로 한다.

### SR-03 산란 모델(근사 물리)

-   SR-03.1 centroid 산란 모델을 baseline으로 제공한다.
-   SR-03.2 facet/patch 기반 산란 + BRDF(Lambert + Phong) 옵션을
    제공한다. 기본 계수: α=0.7(diffuse), β=0.3(specular), p=10(Phong exponent).
-   SR-03.3 Shadow mask(LOS 근사)를 제공한다. 기본 알고리즘: height-profile test.
-   SR-03.4 Layover 근사(누적 허용/flag)를 제공한다.
-   SR-03.5 (옵션) 2-bounce(코너 반사) 경로 누적을 제공한다(후보 집합
    제한 필수).

### SR-04 Raw 데이터 생성

-   SR-04.1 scatterer list로부터 raw echo를 생성할 수 있어야 한다.
-   SR-04.2 time-domain IQ를 기본(default)으로 지원한다. freq-domain은 옵션.
-   SR-04.3 에너지 붕괴(≈0) 방지를 위해 자동 스케일/게인 정규화 옵션을
    제공한다.

### SR-04b 플랫폼 궤적 생성

-   SR-04b.1 합성 직선 궤적(stripmap)을 생성할 수 있어야 한다.
-   SR-04b.2 궤적 파라미터: 고도, 속도, 방위각, 룩 방향을 config에서 지정 가능해야 한다.

### SR-05 이미징

-   SR-05.1 BP 이미징을 기본으로 제공한다.
-   SR-05.2 BP는 타일링(ROI split + overlap + stitch)을 지원해야 한다.
    overlap 기본값: 타일 크기의 10% (최소 8 pixel).
-   SR-05.3 BP는 VRAM 상한을 지키기 위한 streaming(K_batch, M_tile)을
    지원해야 한다.

### SR-06 제품/지표/로그

-   SR-06.1 SLC(복소) 및 intensity 제품을 저장한다(NPY + PNG 필수, GeoTIFF 옵션).
-   SR-06.2 품질지표(SNR/PSLR/ISLR)는 필수.
-   SR-06.3 런타임/VRAM/타일 설정/seed/git hash를 결과 폴더에 기록한다.

### SR-07 시각적 검증 (신규)

-   SR-07.1 BP 복원 후 자동 진단 플롯을 생성해야 한다: intensity(dB), phase,
    range/azimuth cut, dynamic range histogram.
-   SR-07.2 점 타겟 장면에서는 IRF 2D contour + range/azimuth 1D cut 플롯을
    자동 생성해야 한다.
-   SR-07.3 도시 장면에서는 shadow/layover 영역이 표시된 오버레이 플롯을
    생성해야 한다.
-   SR-07.4 시각적 검증 결과는 `results/<run_id>/visual_report/`에 PNG로 저장한다.

### SR-08 실제 도심 데이터 지원 (v1.2 신규)

-   SR-08.1 실제 항공사진/위성영상 TIF 파일을 입력으로 지원해야 한다.
    CRS가 없는 TIF도 처리 가능(기본 pixel spacing 추정).
-   SR-08.2 TIF→ENU 좌표 정합: TIF의 행 방향(row 0=top/north)을
    ENU의 y 방향(row 0=south)으로 자동 변환(수직 플립).
-   SR-08.3 텍스처 기반 pseudo-DEM 생성: 밝기 기반 건물 감지(임계값 + 형태학적 정리).
-   SR-08.4 텍스처 기반 RCS 맵 생성: 밝기→RCS 선형 매핑 + Sobel 엣지 부스팅.
-   SR-08.5 텍스처-SAR 비교 오버레이 플롯을 자동 생성하여 좌표 정합 확인.

### SR-09 스페클 노이즈 (v1.2 신규)

-   SR-09.1 BP 복원 후 multiplicative speckle noise를 추가하는 옵션을 제공한다.
-   SR-09.2 single-look speckle (Rayleigh distributed amplitude) 기본,
    multi-look ENL 조절 가능.
-   SR-09.3 speckle 시드는 config seed와 연동하여 재현 가능해야 한다.

### SR-10 성능 프로파일링 (v1.2 신규)

-   SR-10.1 각 파이프라인 스테이지별 실행 시간을 자동 측정/기록해야 한다.
-   SR-10.2 성능 요약(stage_timings)을 metrics.json에 포함해야 한다.

------------------------------------------------------------------------

## 4. 성능/리소스 요구사항(Performance Requirements)

### PR-01 VRAM 제한 준수

-   PR-01.1 `config.bp.vram_limit_gb` 이하로 동작해야 한다(예: 8GB).
-   PR-01.2 VRAM 측정은 CuPy mempool 기준. GPU 미사용 시 NumPy fallback에서는 RAM 기준.

### PR-02 규모별 실행 목표(예시)

-   PR-02.1 512×512 ROI: 1분 이내(개발 PC 기준; 사용자 조정 가능)
-   PR-02.2 1024×1024 ROI: 5분 이내
-   PR-02.3 2048×2048 ROI: 20분 이내
    > 위 시간은 "목표"이며 HW에 따라 조정 가능. GPU init 시간 제외, 이미징
    > 단계만 측정. 핵심은 **타일/배치 증가 시 선형 스케일**을 유지하는 것.

------------------------------------------------------------------------

## 5. 검증 가능 수락 기준(Acceptance Criteria)

-   AC-01 (IRF) 점 타겟에서 PSLR ≤ -13 dB, ISLR ≤ -10 dB.
    측정법: 4x zero-padding 후 peak 기준 range/azimuth 1D cut에서 mainlobe/sidelobe 분리.
-   AC-02 (코너) 2-bounce on에서 코너 peak가 off 대비 ≥ +6 dB.
    Peak 탐지: 코너 예상 위치 ±3 pixel 범위 내 max.
-   AC-03 (그림자) shadow 영역 평균 intensity가 non-shadow 대비 ≤ -15 dB.
    영역 정의: vis_mask 기반(shadow: vis_mask < 0.5, non-shadow: vis_mask ≥ 0.5).
-   AC-04 (정합) ROI 내부 상대 위치 오차 < 1 pixel.
-   AC-05 (강건성) 파라미터 스윕에서 신호 붕괴 비율 ≤ 5%.
-   AC-06 (성능) 2048² ROI가 VRAM 상한 내 실행 + 스케일 편차 ±20% 이내.
-   AC-07 (시각적) BP 결과 이미지에서 다음을 자동 판정:
    (a) dynamic range ≥ 30 dB, (b) 점 타겟 PSF 대칭성 (aspect ratio ≤ 1.5),
    (c) 도시 장면에서 shadow 영역이 시각적으로 구분 가능.
-   AC-08 (실제 데이터) 실제 도심 TIF 입력 시:
    (a) 텍스처-SAR 오버레이에서 건물 위치 대응 확인,
    (b) shadow 방향이 레이더 룩 방향과 일관,
    (c) dynamic range ≥ 40 dB.

------------------------------------------------------------------------

## 6. 산출물(Deliverables)

-   요구사항 문서(본 문서)
-   단위기능 목록 및 데이터 계약(UF spec)
-   단위기능 테스트 문서(각 UF별 실험/체크리스트)
-   통합 설계(파이프라인/e2e.py 관점) + 구성 파일 스키마
-   통합 검증(실험 설계 + CI 회귀 기준)
-   **시각적 검증 리포트(BP 결과 진단 플롯 모음)**
