"""Integration test for EXP-07: Urban E2E visual verification.

Verifies AC-03 (shadow contrast), AC-07 (visual metrics), visual report generation.
"""
import numpy as np
import os
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sar_imaging.pipeline.e2e import run_pipeline


def test_exp07_urban_visual():
    """EXP-07: Urban scene should show building structure, shadow, layover."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                'configs', 'exp', 'exp07_urban_visual.yaml')

    results = run_pipeline(config_path=config_path,
                            config_override={
                                'output': {'run_id': 'test_exp07', 'root_dir': 'results'},
                            })

    metrics = results['metrics']
    visual = results.get('visual_metrics', {})

    # AC-07: Dynamic range ≥ 30 dB
    dr = visual.get('dynamic_range_db', metrics.get('dynamic_range_db', 0))
    assert dr >= 30.0, f"AC-07 FAIL: Dynamic range = {dr:.1f} dB < 30 dB"

    # AC-03: Shadow contrast ≤ -15 dB
    shadow_contrast = visual.get('shadow_contrast_db', 0)
    assert shadow_contrast <= -15.0, (
        f"AC-03 FAIL: Shadow contrast = {shadow_contrast:.1f} dB > -15 dB"
    )

    # Check visual report files exist
    vis_dir = os.path.join(results['output_dir'], 'visual_report')
    expected_files = ['intensity_db.png', 'phase.png', 'histogram.png',
                      'shadow_overlay.png', 'layover_overlay.png']
    for fname in expected_files:
        fpath = os.path.join(vis_dir, fname)
        assert os.path.exists(fpath), f"Missing visual report: {fpath}"

    # Check products saved
    assert os.path.exists(os.path.join(results['output_dir'], 'slc.npy'))
    assert os.path.exists(os.path.join(results['output_dir'], 'metrics.json'))
    assert os.path.exists(os.path.join(results['output_dir'], 'config.yaml'))

    print(f"  PASS: EXP-07 DR={dr:.1f} dB, shadow_contrast={shadow_contrast:.1f} dB")


if __name__ == '__main__':
    test_exp07_urban_visual()
    print("EXP-07 integration test passed.")
