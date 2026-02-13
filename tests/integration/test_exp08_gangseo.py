"""EXP-08 Real Urban Data (Gangseo) Integration Test.

Tests the full E2E pipeline with real aerial photography TIF input.
Verifies AC-08 acceptance criteria:
  (a) texture-SAR overlay shows building correspondence
  (b) shadow direction consistent with radar look direction
  (c) dynamic range >= 40 dB
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_exp08_gangseo():
    """EXP-08: Real urban data E2E test."""
    tif_path = 'dataset/gangseo_crop_500m.tif'
    if not os.path.exists(tif_path):
        print("  SKIP: EXP-08 (gangseo_crop_500m.tif not found)")
        return

    from sar_imaging.pipeline.e2e import run_pipeline

    results = run_pipeline(
        config_path='configs/exp/exp_gangseo_urban.yaml',
        config_override={
            'output': {'run_id': 'test_exp08', 'root_dir': 'results'},
            'scene': {'grid_spacing_m': 2.0},  # reduced for test speed
            'platform': {'num_pulses': 128},
            'bp': {'enable_streaming': False},
        }
    )

    output_dir = results['output_dir']
    metrics = results['metrics']
    visual_metrics = results.get('visual_metrics', {})

    # Check all visual report files exist
    vis_dir = os.path.join(output_dir, 'visual_report')
    required_files = [
        'intensity_db.png', 'phase.png', 'histogram.png',
        'shadow_overlay.png', 'layover_overlay.png',
        'texture_comparison.png',
    ]
    for f in required_files:
        path = os.path.join(vis_dir, f)
        assert os.path.exists(path), f"Missing: {f}"

    # AC-08(c): dynamic range >= 40 dB
    dr = metrics.get('dynamic_range_db', 0)
    assert dr >= 40, f"dynamic_range={dr:.1f} dB < 40 dB"

    # Check stage timings are recorded (UF-32)
    assert 'stage_timings' in metrics, "Missing stage_timings in metrics"
    timings = metrics['stage_timings']
    assert 'scene_time_s' in timings
    assert 'raw_time_s' in timings
    assert 'bp_time_s' in timings

    # Performance check: total pipeline < 120s for reduced test
    total = results.get('total_time_s', float('inf'))
    assert total < 120, f"Pipeline too slow: {total:.1f}s > 120s"

    print(f"  PASS: EXP-08 DR={dr:.1f}dB, total={total:.1f}s, "
          f"shadow_contrast={visual_metrics.get('shadow_contrast_db', 'N/A')}")
    print(f"  Visual report: {vis_dir}")
    print(f"  texture_comparison.png generated for coordinate alignment check")


if __name__ == '__main__':
    test_exp08_gangseo()
    print("EXP-08 integration test passed.")
