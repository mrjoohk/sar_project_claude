"""CLI tool to run SAR experiments.

Usage:
    python -m tools.run_exp --config configs/exp/exp01_point_target.yaml
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sar_imaging.pipeline.e2e import run_pipeline


def main():
    parser = argparse.ArgumentParser(description='Run SAR experiment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config YAML')
    args = parser.parse_args()

    results = run_pipeline(config_path=args.config)

    print("\n=== Results Summary ===")
    print(f"Output: {results['output_dir']}")
    print(f"Total time: {results.get('total_time_s', 0):.2f}s")

    metrics = results.get('metrics', {})
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")

    visual = results.get('visual_metrics', {})
    for key, val in visual.items():
        if isinstance(val, float):
            print(f"  visual.{key}: {val:.3f}")


if __name__ == '__main__':
    main()
