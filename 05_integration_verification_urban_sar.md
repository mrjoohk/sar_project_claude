# Urban SAR 재수행 패키지 5/5: 통합 검증(Integration Verification) (v1.2, 2026-02-11)

본 문서는 UF 통합 후 E2E 수준에서 요구사항을 만족하는지 검증한다.
EXP-01\~09을 **CI/회귀 가능한 형태**로 구체화한다.

------------------------------------------------------------------------

## 공통 규칙

-   모든 실험은 `config.yaml`로 재현 가능해야 한다.
-   실험마다 `metrics.json`과 최소 1개 이미지 산출물을 저장한다.
-   **실험마다 `visual_report/` 진단 플롯을 생성하고 에이전트가 시각 확인한다.**
-   CI에서는 "축소 버전(작은 ROI, 작은 K)"을 수행하고, 오프라인에서
    full-scale을 수행한다.

------------------------------------------------------------------------

# EXP-01 IRF(점 타겟) 기준

-   입력: 평탄 DEM + 점타겟 1개 (scene center)
-   파이프라인: BP(타일 off) vs BP(타일 on)
-   수락:
    -   PSLR ≤ -13 dB, ISLR ≤ -10 dB
    -   타일링 on/off peak-normalized RMS < 1e-3
-   **시각적 검증:**
    -   에이전트가 `irf_2d_contour.png` 확인: PSF가 중심에 위치, 대칭 sidelobe
    -   에이전트가 `range_cut.png` / `azimuth_cut.png` 확인: mainlobe 폭이 이론 해상도와 일관

# EXP-02 코너 반사(2-bounce)

-   입력: L자 코너(두 벽) + 지면
-   파이프라인: bounce depth=1 vs 2
-   수락:
    -   코너 peak가 bounce=1 대비 ≥ +6 dB
    -   룩각 스윕(최소 3단계)에서 peak 위치 변동 < 2 pixel
-   **시각적 검증:**
    -   에이전트가 `intensity_db.png` 확인: 코너 위치에 밝은 점 존재
    -   bounce=1 vs bounce=2 이미지 비교: 코너 영역 밝기 증가 시각 확인

# EXP-03 Shadow

-   입력: 높은 벽 구조 (높이 30m, 폭 5m)
-   파이프라인: shadow off vs on
-   수락:
    -   shadow 영역 평균 intensity가 non-shadow 대비 ≤ -15 dB
    -   룩각 변화(3단계) 시 shadow 길이가 단조 증가/감소
        (incidence angle 증가 → shadow 길이 증가, 길이 변화율 > 10%)
-   **시각적 검증:**
    -   에이전트가 `shadow_overlay.png` 확인: shadow가 벽 반대편에 위치
    -   에이전트가 `intensity_db.png` 확인: shadow 영역이 어두운 영역으로 명확히 구분

# EXP-04 좌표/정합

-   입력: known landmark(점/코너) 좌표 리스트 (최소 4개)
-   파이프라인: UF-22 on
-   수락:
    -   상대 위치 오차 < 1 pixel
    -   (옵션) UF-23 on일 때 절대 RMSE < 2 pixels

# EXP-05 강건성(민감도)

-   입력: 도시 블록 장면
-   파이프라인: UF-24/25 on
-   파라미터 스윕: beam gate 3단계, rcs scale ±20 dB
-   수락:
    -   신호 붕괴(에너지≈0) 비율 ≤ 5%
    -   SNR 표준편차 30% 이상 감소(자동 파라미터 사용 시)

# EXP-06 성능/메모리(Streaming BP)

-   입력: ROI 512²/1024²/2048², K 스윕
-   파이프라인: UF-27 on, vram_limit_gb 설정
-   수락:
    -   메모리 상한 초과 금지(로그)
    -   면적 증가에 따른 런타임 스케일 편차 ±20%

# EXP-07 시각적 E2E 검증 (신규)

-   입력: 도시 블록 장면 (box DEM + shadow + layover)
-   파이프라인: 전체 E2E (scene → raw → BP → metrics → visual)
-   수락:
    -   모든 visual_report 파일 생성됨
    -   dynamic_range_db ≥ 30
    -   **에이전트 시각 판정:**
        1. `intensity_db.png`: 건물 구조가 시각적으로 구분됨
        2. `shadow_overlay.png`: shadow가 건물 반대편에 올바르게 위치
        3. `histogram.png`: bimodal 분포 (건물 밝음 + shadow 어두움)
        4. `layover_overlay.png`: layover가 건물 레이더 방향 면에 위치

# EXP-08 실제 도심 데이터 E2E (v1.2 신규)

-   입력: 실제 항공사진 TIF (강서구 500m×500m crop)
-   파이프라인: texture_path → UF-30 → 전체 E2E
-   수락:
    -   모든 visual_report 파일 생성됨
    -   `texture_comparison.png` 생성됨
    -   dynamic_range_db ≥ 40
    -   **에이전트 시각 판정 (AC-08):**
        1. `texture_comparison.png`: 건물 위치가 SAR 영상과 공간적으로 대응
        2. `shadow_overlay.png`: shadow 방향이 레이더 룩 방향(negative x → positive x)과 일관
        3. `intensity_db.png`: 건물 엣지에서 밝은 반사, 도로에서 낮은 반사
        4. 좌표 정합: TIF bottom-left가 ENU y_min, x_min에 대응

# EXP-09 스페클 노이즈 검증 (v1.2 신규)

-   입력: 도시 블록 장면 (box DEM)
-   파이프라인: speckle off vs on (ENL=1), 동일 seed
-   수락:
    -   speckle off: ENL > 10 (깨끗한 영상)
    -   speckle on (ENL=1): measured ENL ≈ 1.0 (±0.5)
    -   speckle on 시 intensity histogram이 exponential-like 분포
    -   **에이전트 시각 판정:**
        1. speckle off 영상 대비 on 영상에서 granular noise 패턴 확인
        2. 건물 구조는 여전히 시각적으로 구분 가능

------------------------------------------------------------------------

## CI 회귀 규칙

-   PSLR/ISLR 악화 > 1 dB → fail
-   geo RMSE 악화 > 0.5 pixel → fail
-   runtime 악화 > 25% → warn/fail
-   dynamic_range_db < 25 → fail
-   visual_report 파일 누락 → fail

------------------------------------------------------------------------

## 권장 결과 폴더 구조

-   `results/EXP-xx/`
    -   `config.yaml`
    -   `slc.npy`, `intensity.png`
    -   `metrics.json`
    -   `visual_report/`
        -   `intensity_db.png`
        -   `phase.png`
        -   `histogram.png`
        -   `range_cut.png`, `azimuth_cut.png` (점 타겟)
        -   `irf_2d_contour.png` (점 타겟)
        -   `shadow_overlay.png` (도시)
        -   `layover_overlay.png` (도시)
        -   `texture_comparison.png` (실제 데이터, v1.2)
    -   `visual_metrics.json`
    -   `log.txt`
