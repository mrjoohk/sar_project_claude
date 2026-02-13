"""Configuration loader and validator for the SAR pipeline.

Implements Stage-0 Config Validation from 04_integration_design_urban_sar.md.
"""
import yaml
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {
    'experiment.id': str,
    'seed': int,
    'radar.fc_hz': (int, float),
    'radar.bandwidth_hz': (int, float),
    'radar.prf_hz': (int, float),
    'radar.tp_s': (int, float),
    'platform.altitude_m': (int, float),
    'platform.velocity_m_s': (int, float),
    'platform.look_angle_deg': (int, float),
    'imaging.method': str,
}

DEFAULTS = {
    'experiment.id': 'default',
    'experiment.description': '',
    'seed': 42,
    'output.run_id': 'run_001',
    'output.root_dir': 'results',

    # Scene
    'scene.build_in_e2e': True,
    'scene.synthetic.type': 'flat',
    'scene.synthetic.size': [512, 512],
    'scene.synthetic.spacing_m': 1.0,
    'scene.synthetic.box_height_m': 30.0,
    'scene.synthetic.box_width_m': 50.0,
    'scene.point_targets': [],

    # Radar (X-band airborne-like defaults)
    'radar.fc_hz': 9.65e9,
    'radar.bandwidth_hz': 300e6,
    'radar.prf_hz': 500.0,
    'radar.tp_s': 10e-6,

    # Platform
    'platform.altitude_m': 5000.0,
    'platform.velocity_m_s': 100.0,
    'platform.look_angle_deg': 45.0,
    'platform.azimuth_deg': 0.0,
    'platform.num_pulses': 256,

    # Imaging
    'imaging.method': 'bp',

    # Scatter
    'scatter.model': 'facet_brdf',
    'scatter.enable_shadow': True,
    'scatter.enable_layover': True,
    'scatter.enable_adaptive_threshold': False,
    'scatter.enable_power_norm': True,
    'scatter.multibounce.depth': 1,
    'scatter.brdf.alpha': 0.7,
    'scatter.brdf.beta': 0.3,
    'scatter.brdf.p': 10.0,

    # BP
    'bp.enable_streaming': False,
    'bp.vram_limit_gb': 8.0,
    'bp.tile_size': [128, 128],
    'bp.k_batch': 64,
    'bp.overlap': 16,
    'bp.stitch_method': 'linear_blend',

    # Coordinate
    'geo.enable_unified_enu': True,

    # Visual
    'visual.enable': True,
    'visual.scene_type': 'point_target',

    # Logging
    'logging.level': 'INFO',
}


def _get_nested(d: dict, key: str, default=None):
    """Get nested dict value using dot notation."""
    parts = key.split('.')
    current = d
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _set_nested(d: dict, key: str, value):
    """Set nested dict value using dot notation."""
    parts = key.split('.')
    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str = None) -> Dict[str, Any]:
    """Load and validate config from YAML file.

    If path is None, returns defaults.
    Preserves ALL keys from the YAML, applying defaults only for missing keys.
    """
    if path and os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # Build defaults as nested dict
    defaults_nested = {}
    for key, default_val in DEFAULTS.items():
        _set_nested(defaults_nested, key, default_val)

    # Merge: user config overrides defaults, preserving all user keys
    result = _deep_merge(defaults_nested, cfg)

    return result


def _coerce_numerics(cfg: dict):
    """Coerce string values that should be numeric (YAML float parsing issue)."""
    numeric_keys = [k for k, t in REQUIRED_KEYS.items()
                    if t in ((int, float), float, int)]
    for key in numeric_keys:
        val = _get_nested(cfg, key)
        if isinstance(val, str):
            try:
                _set_nested(cfg, key, float(val))
            except ValueError:
                pass


def validate_config(cfg: dict):
    """Validate config schema. Raises ValueError on failure."""
    _coerce_numerics(cfg)
    errors = []
    for key, expected_type in REQUIRED_KEYS.items():
        val = _get_nested(cfg, key)
        if val is None:
            errors.append(f"Missing required key: {key}")
        elif not isinstance(val, expected_type):
            errors.append(f"Key '{key}' has type {type(val).__name__}, expected {expected_type}")

    method = _get_nested(cfg, 'imaging.method')
    if method not in ('bp', 'standard', 'hybrid'):
        errors.append(f"imaging.method must be bp/standard/hybrid, got '{method}'")

    model = _get_nested(cfg, 'scatter.model')
    if model not in ('centroid', 'facet_brdf'):
        errors.append(f"scatter.model must be centroid/facet_brdf, got '{model}'")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    logger.info("Config validation passed.")
