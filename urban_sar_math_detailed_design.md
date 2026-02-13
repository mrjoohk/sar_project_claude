# Urban SAR 수학 모델 상세 설계 문서 (v1.0)

## 1. 시스템 입력

### 1.1 입력 데이터

-   DEM (GeoTIFF)
-   Texture image
-   Radar parameters:
    -   Center frequency f_c
    -   Bandwidth B
    -   PRF
    -   Platform trajectory (state vectors)

------------------------------------------------------------------------

## 2. 산란 모델

### 2.1 Facet 기반 RCS 모델

Facet normal:\
n = (v2 - v1) × (v3 - v1)

입사각: cos(θ_i) = n · s_hat

RCS 모델: σ = A \[ α cos(θ_i) + β \|r_hat · s_hat\|\^p \]

where: - A: facet area - α, β: weighting coefficients - p: specular
exponent

출력: σ_i for each facet

------------------------------------------------------------------------

## 3. Raw IQ 생성

거리: R_i(t) = \|\| x_platform(t) - x_i \|\|

위상: φ_i(t) = -4π f_c R_i(t) / c

수신 신호: s(t) = Σ σ_i exp(j φ_i(t))

------------------------------------------------------------------------

## 4. Backprojection

이미지 픽셀 x:

I(x) = Σ s(t_k) exp(j 4π f_c R(x, t_k)/c)

------------------------------------------------------------------------

## 5. 출력

-   Complex SAR image
-   Geocoded image
-   Quality metrics (SNR, PSLR, ISLR, ENL)

------------------------------------------------------------------------

## 6. 제한사항

-   Single-bounce scattering only
-   No full EM simulation
-   Layover approximated
-   Shadow simplified LOS test
