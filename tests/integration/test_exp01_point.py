"""Integration test for EXP-01: IRF point target baseline.

Verifies AC-01 (PSLR/ISLR) and AC-07 (visual metrics).
"""
import numpy as np
import os
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sar_imaging.pipeline.e2e import run_pipeline


def test_exp01_point_target():
    """EXP-01: Point target IRF should meet AC-01 and AC-07."""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                'configs', 'exp', 'exp01_point_target.yaml')

    results = run_pipeline(config_path=config_path,
                            config_override={
                                'output': {'run_id': 'test_exp01', 'root_dir': 'results'},
                            })

    metrics = results['metrics']

    # AC-01: PSLR ≤ -13 dB
    pslr = metrics['pslr_db']
    assert pslr <= -13.0, f"AC-01 FAIL: PSLR = {pslr:.1f} dB > -13 dB"

    # AC-01: ISLR ≤ -10 dB
    islr = metrics['islr_db']
    assert islr <= -10.0, f"AC-01 FAIL: ISLR = {islr:.1f} dB > -10 dB"

    # AC-07: Dynamic range ≥ 30 dB
    visual = results.get('visual_metrics', {})
    dr = visual.get('dynamic_range_db', metrics.get('dynamic_range_db', 0))
    assert dr >= 30.0, f"AC-07 FAIL: Dynamic range = {dr:.1f} dB < 30 dB"

    # AC-07: PSF aspect ratio ≤ 1.5
    ar = visual.get('psf_aspect_ratio', 1.0)
    assert ar <= 1.5, f"AC-07 FAIL: PSF aspect ratio = {ar:.2f} > 1.5"

    # Check visual report files exist
    vis_dir = os.path.join(results['output_dir'], 'visual_report')
    expected_files = ['intensity_db.png', 'phase.png', 'histogram.png',
                      'irf_2d_contour.png', 'range_cut.png', 'azimuth_cut.png']
    for fname in expected_files:
        fpath = os.path.join(vis_dir, fname)
        assert os.path.exists(fpath), f"Missing visual report: {fpath}"

    print(f"  PASS: EXP-01 PSLR={pslr:.1f} dB, ISLR={islr:.1f} dB, "
          f"DR={dr:.1f} dB, AR={ar:.2f}")


if __name__ == '__main__':
    test_exp01_point_target()
    print("EXP-01 integration test passed.")
