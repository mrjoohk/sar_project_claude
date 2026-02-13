# Claude Code Project Guide — Urban SAR Simulator/Imager

## 0) Project Identity
You are working on an Urban SAR Simulator/Imager project driven by documents:
- `01_requirements_urban_sar.md`
- `02_unit_functions_urban_sar.md`
- `03_unit_verification_urban_sar.md`
- `04_integration_design_urban_sar.md`
- `05_integration_verification_urban_sar.md`
- `urban_sar_e2e_pipeline_integrated_design.md`
- math + cuda + complexity + specs + verification docs (v1.1/v2)

**Goal**
Implement a reproducible, testable pipeline:
Requirements → Unit Functions (UF) → UF tests → Integration → Integration verification (EXP-01~06)

**Non-goals (v1)**
- Full-wave EM (MoM/FDTD/PO), absolute radiometric calibration, multi-GPU distribution.

---

## 1) Operating Rules (VERY IMPORTANT)

### 1.1 Deliverables-first
Every change MUST map to:
- a specific requirement (SR/PR/AC in `01_requirements_urban_sar.md`), AND
- a specific UF and/or test (`02_unit_functions_urban_sar.md`, `03_unit_verification_urban_sar.md`, `05_integration_verification_urban_sar.md`).

### 1.2 No ambiguity
For every implementation task, you MUST provide:
- Inputs / Outputs / Constraints
- Deterministic acceptance criteria
- How to run the test (exact command)

### 1.3 Minimal working increments (priority order)
1) UF-22 + UF-27 (alignment + streaming BP) to stabilize geometry/perf
2) UF-19 (facet/BRDF)
3) UF-20, UF-21 (shadow/layover)
4) UF-24, UF-25 (robustness)
5) UF-26 (metrics extension)
Then wire EXP-01~06.

### 1.4 Reproducibility required
Every run must save:
- `config.yaml`
- `meta.json` including: git hash, seed, timestamp
- products: `slc.npy` + `intensity.png` (or GeoTIFF)
- `metrics.json`
- (optional) `profile.json`
in `results/<run_id>/`

### 1.5 Don’t “guess” physics
If something is unclear:
- read the referenced md docs first,
- propose a minimal consistent assumption,
- encode that assumption into config with default values,
- add a test.

---

## 2) Repository Layout (target structure)
Create/maintain the following structure:

```
sar_imaging/
  pipeline/
    e2e.py
  uf/
    uf01_ingest.py
    ...
    uf27_streaming_bp.py   # or inside uf09_bp.py
  scene/
    dem_loader.py
    texture_align.py
    scatterers.py          # UF-19~25 placement if build_in_e2e=True
  metrics/
    uf17_metrics.py
    uf26_metrics_ext.py
  cuda/
    kernels/
      bp_streaming.cu
      raw_forward.cu
    compile.py
configs/
  exp/
tests/
  unit/
  integration/
tools/
  run_exp.py
  make_report.py
```

If current repo differs, create thin adapter layers rather than rewriting everything.

---

## 3) Config Schema (must exist)
Use a single YAML schema (recommend `configs/exp/*.yaml`).

Required keys (minimum):
- `experiment.id`
- `experiment.description`
- `seed`
- `output.run_id`
- `output.root_dir`
- `scene.build_in_e2e`
- `scene.dem_path`
- `scene.texture_path`
- `scene.roi_enu` (xmin,xmax,ymin,ymax) in meters
- `scene.grid.spacing_m` (pixel spacing)
- `imaging.method: bp|standard|hybrid`
- `geo.enable_unified_enu`
- `geo.enable_geocode`
- `geo.enable_orthorectify`
- `scatter.model: centroid|facet_brdf`
- `scatter.enable_shadow`
- `scatter.enable_layover`
- `scatter.enable_adaptive_threshold`
- `scatter.enable_power_norm`
- `scatter.multibounce.depth: 0|1|2`
- `bp.enable_streaming`
- `bp.vram_limit_gb`
- `bp.tile_size: [Nx, Ny]`
- `bp.k_batch`
- `bp.overlap`
- `bp.stitch_method`
- `radar.fc_hz`
- `radar.bandwidth_hz`
- `radar.prf_hz`
- `radar.tp_s`
- `platform.mode` and trajectory parameters (or a path to state vectors)

All defaults must be explicitly written in each experiment template.

---

## 4) Data Contracts (must be enforced)

### 4.1 Meta
`meta.platform`: `t_k`, `p_k(x,y,z)`, `v_k`, `attitude_k`  
`meta.radar`: `fc`, `B`, `PRF`, `Tp`, optional `f_n`  
`meta.geo`: `crs`, `enu_origin`, `transforms`

### 4.2 Scatterers (SoA, GPU-friendly)
Required arrays (float32):
- `sx, sy, sz, s_rcs`
Optional:
- `snx, sny, snz, s_area, mat_id, vis_mask, layover_flag`

### 4.3 Raw
- time-domain: `raw[K, Nr] complex64`
- freq-domain: `raw[K, Nf] complex64`

### 4.4 BP Grid
- `bp_grid_xyz[M,3] float32` or SoA `px,py,pz`

Add explicit validators. If validation fails, fail fast with actionable errors.

---

## 5) Testing Strategy (required)

### 5.1 Unit tests (UF-level)
Implement tests matching `03_unit_verification_urban_sar.md`.
Minimum required UF tests for first milestone:
- UF22-T01 (round-trip ENU/CRS error)
- UF27-T01 (tiling on/off equivalence on point target)
- UF27-T02 (VRAM limit respected / logged)

Then expand to:
- UF19-T01/T02
- UF20-T01
- UF21-T01
- UF24-T01/T02
- UF25-T01
- UF26-T01/T02

### 5.2 Integration tests (EXP-level)
Implement EXP-01~06 matching `05_integration_verification_urban_sar.md`.
CI must run at least:
- EXP-01 (small ROI)
- EXP-04 (small landmarks)
- EXP-06 (small ROI, VRAM logging)

### 5.3 Acceptance criteria (regression gates)
Hard fail if:
- PSLR/ISLR regresses > 1 dB
- geo RMSE regresses > 0.5 pixel
- runtime regresses > 25% (warn/fail policy)

---

## 6) GPU/CUDA Rules (CuPy RawKernel)
- Keep CUDA kernels in `sar_imaging/cuda/kernels/*.cu`.
- Use NVRTC via CuPy `RawKernel`. Avoid arch-specific flags.
- Prefer SoA, coalesced access, chunking/shared memory.
- Avoid atomics per pixel; reduce per block then atomic once.
- Log VRAM usage and kernel time per batch.
- Provide fast-math toggle vs accurate-math.

---

## 7) Implementation Milestones (do in order)

### Milestone A — Geometry + Performance baseline
- UF-22 (unified ENU)
- UF-27 (streaming BP) within UF-09(method="bp") or UF-09a
- Unit tests: UF22-T01, UF27-T01, UF27-T02
- Integration test: EXP-01 (point target)

### Milestone B — Urban physics minimal realism
- UF-19 (facet/BRDF Lambert+Phong)
- UF-20 shadow mask (height test first)
- UF-21 layover flag/weight
- Integration tests: EXP-02, EXP-03

### Milestone C — Robustness + Metrics
- UF-24 adaptive threshold
- UF-25 power normalization
- UF-26 metrics extension
- Integration test: EXP-05

### Milestone D — Scale & regression
- Full EXP-06 profiling and scaling runs
- CI regression gates

---

## 8) How to Work (Claude Code interaction protocol)
When asked to implement something:
1) Quote the exact requirement IDs and UF IDs you’re addressing.
2) Propose file changes (list of files).
3) Provide an implementation plan in <= 10 bullets.
4) Implement.
5) Add/Update tests.
6) Show exact commands to run:
   - unit test
   - integration exp
7) Summarize results and any known limitations.

Do NOT skip tests.

---

## 9) Default Commands (must exist)
Provide these CLI entrypoints (or equivalent):
- `python -m tools.run_exp --config configs/exp/exp01_point.yaml`
- `pytest -q tests/unit/test_uf22_*.py`
- `pytest -q tests/unit/test_uf27_*.py`
- `pytest -q tests/integration/test_exp01_*.py`

---

## 10) Notes on “Realistic SAR Specs”
Use parameter presets (spaceborne-like / airborne-like) from the specs doc.
Do not claim exact real-sensor values; keep them as ranges/presets.

---

## 11) Definition of Done (global)
A PR is “done” only if:
- All modified UFs have unit tests
- At least one EXP integration test passes
- Results artifacts are saved with config/meta/metrics
- VRAM usage is logged for BP runs
- No acceptance criteria regressions
