# Urban SAR Simulator / Imager 개선 로드맵 (v1.0)

## 0. Baseline 정의

현재 시스템은 다음 기능을 포함한다:

-   DEM/GeoTIFF 기반 장면 구성
-   Centroid 기반 산란체 생성
-   Texture 기반 RCS (variance / edge)
-   Raw IQ 생성
-   BP 기반 이미징 (타일링 지원)
-   품질지표(SNR/PSLR/ISLR)

현재 상태는: \> Urban PoC SAR Imaging Framework

------------------------------------------------------------------------

## 1. 개선 목표

  Phase     목표
  --------- -------------------------------
  Phase 1   도심 산란 물리 현상 최소 재현
  Phase 2   좌표 및 정합 안정화
  Phase 3   영상 품질 안정화
  Phase 4   GPU 최적화 및 확장성 확보

------------------------------------------------------------------------

# Phase 1 --- 산란 모델 고도화

## UF-19: Facet 기반 산란 모델

### 목적

Centroid 산란 모델을 facet 기반 물리 모델로 확장

### 요구사항

-   DEM mesh triangle 유지
-   facet normal 계산
-   입사각 계산
-   Lambertian + Specular 혼합 모델 적용

### 출력

scatterer list: {xyz, normal, area, rcs}

------------------------------------------------------------------------

## UF-20: Shadow Masking

### 요구사항

-   플랫폼 → scatterer LOS 계산
-   DEM 기반 차폐 판별
-   차폐 시 amplitude = 0

------------------------------------------------------------------------

## UF-21: Layover 근사

### 요구사항

-   고도 gradient 기반 layover flag
-   동일 range bin 다중 scatterer 허용

------------------------------------------------------------------------

# Phase 2 --- 좌표 통합

## UF-22: Unified Scene Coordinate System

### 요구사항

-   GeoTIFF CRS → local ENU 변환
-   모든 데이터 동일 좌표계 사용
-   역변환 오차 \< 1 pixel

------------------------------------------------------------------------

# Phase 3 --- 품질 안정화

## UF-24: Adaptive Scatterer Threshold

-   percentile 기반 자동 threshold

## UF-25: Beam/Power Normalization

-   거리 감쇠 자동 보정

------------------------------------------------------------------------

# Phase 4 --- GPU 최적화

## UF-27: Streaming BP

-   batch-wise azimuth streaming
-   VRAM usage configurable

------------------------------------------------------------------------

최종 목표: \> Physically Consistent Urban SAR Simulation Framework
