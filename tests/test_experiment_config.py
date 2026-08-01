from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_config import build_run_manifest, validate_experiment
from src.runtime_profile import OMP_TMP_WARNING, _build_runtime_profile, _filter_native_stderr, apply_runtime_environment
from src.trainer_registry import available_trainers, resolve_trainer
from src.training_loop import run_training_smoke


class ExperimentConfigTests(unittest.TestCase):
    def test_validate_mmb_devsample(self) -> None:
        validation = validate_experiment('v1_mmb_baseline_reproduction', 'devsample')
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['missing_files'], [])
        self.assertEqual(validation['missing_dataset_paths'], [])

    def test_build_run_manifest_contains_expected_seeds(self) -> None:
        manifest = build_run_manifest('v1_mmb_baseline_reproduction', 'viso')
        self.assertEqual(manifest['seeds'], [101, 202, 303, 404, 505])
        self.assertEqual(manifest['dataset']['subset'], 'ship')
        self.assertIn('runtime', manifest)
        self.assertIn(manifest['runtime']['accelerator'], {'cpu', 'cuda', 'mps'})

    def test_materialize_runs_creates_seed_manifests(self) -> None:
        command = [
            sys.executable,
            str(ROOT / 'scripts' / 'materialize_runs.py'),
            '--slug',
            'v1_mmb_baseline_reproduction',
            '--dataset-profile',
            'devsample',
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        self.assertEqual(len(payload['created']), 5)
        self.assertIn('runtime', payload)

    def test_runtime_profile_prefers_mps_on_mac(self) -> None:
        profile = _build_runtime_profile(
            'Darwin',
            {'available': True, 'cuda_available': False, 'mps_available': True},
            False,
            8,
        )
        self.assertEqual(profile.accelerator, 'mps')
        self.assertEqual(profile.device, 'mps')
        self.assertFalse(profile.pin_memory)

    def test_runtime_profile_prefers_cuda_on_nvidia_hosts(self) -> None:
        profile = _build_runtime_profile(
            'Linux',
            {'available': True, 'cuda_available': True, 'mps_available': False},
            True,
            12,
        )
        self.assertEqual(profile.accelerator, 'cuda')
        self.assertTrue(profile.pin_memory)
        self.assertEqual(profile.distributed_backend, 'nccl')

    def test_apply_runtime_environment_sets_env(self) -> None:
        fake_profile = _build_runtime_profile(
            'Darwin',
            {'available': True, 'cuda_available': False, 'mps_available': True},
            False,
            8,
        )
        with mock.patch('src.runtime_profile.detect_runtime_profile', return_value=fake_profile):
            profile = apply_runtime_environment()
        self.assertEqual(profile.device, 'mps')
        self.assertEqual(os.environ['SMALLOO_DEVICE'], 'mps')
        self.assertEqual(os.environ['SMALLOO_ACCELERATOR'], 'mps')

    def test_runtime_profile_keeps_mps_diagnostic_in_notes(self) -> None:
        profile = _build_runtime_profile(
            'Darwin',
            {
                'available': True,
                'cuda_available': False,
                'mps_built': True,
                'mps_available': False,
                'mps_failure_reason': 'RuntimeError: macOS version gate',
            },
            False,
            8,
        )
        self.assertEqual(profile.accelerator, 'cpu')
        self.assertTrue(any('macOS version gate' in note for note in profile.notes))

    def test_mac_runtime_has_explicit_torch_smoke_failure_when_missing(self) -> None:
        if platform.system() != 'Darwin':
            self.skipTest('Mac-specific smoke test')
        command = [
            sys.executable,
            '-c',
            (
                'import importlib.util, platform, sys; '
                'missing = platform.system() == "Darwin" and importlib.util.find_spec("torch") is None; '
                'print("missing_torch_on_mac" if missing else "ok"); '
                'sys.exit(1 if missing else 0)'
            ),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if importlib.util.find_spec('torch') is None:
            self.assertEqual(result.returncode, 1)
            self.assertIn('missing_torch_on_mac', result.stdout)
        else:
            self.assertEqual(result.returncode, 0)

    def test_training_smoke_runs_or_skips_cleanly(self) -> None:
        result = run_training_smoke(seed=101, steps=2)
        self.assertIn(result['status'], {'completed', 'skipped'})
        self.assertEqual(result['seed'], 101)

    def test_filter_native_stderr_drops_known_omp_warning(self) -> None:
        with mock.patch('os.write') as write_mock:
            _filter_native_stderr(OMP_TMP_WARNING + '\n')
        write_mock.assert_not_called()

    def test_filter_native_stderr_preserves_other_messages(self) -> None:
        with mock.patch('os.write') as write_mock:
            _filter_native_stderr('unexpected native warning\n')
        write_mock.assert_called_once()

    def test_trainer_registry_exposes_smoke_and_mmb(self) -> None:
        trainers = available_trainers()
        self.assertIn('smoke', trainers)
        self.assertIn('mmb', trainers)

    def test_mmb_trainer_returns_viso_metrics_payload(self) -> None:
        manifest = build_run_manifest('v1_mmb_baseline_reproduction', 'devsample')
        trainer = resolve_trainer('mmb')
        result = trainer(seed=101, steps=1, manifest=manifest)
        self.assertEqual(result['status'], 'completed')
        self.assertIn('metrics', result)
        self.assertIn('precision', result['metrics'])
        self.assertIn('recall', result['metrics'])
        self.assertIn('f1', result['metrics'])
        self.assertIn('ap', result['metrics'])
        self.assertIn('mAP', result['metrics'])

    def test_manifest_includes_intervention_for_follow_up_versions(self) -> None:
        manifest = build_run_manifest('v2_mmb_tiling_overlap', 'devsample')
        self.assertIn('intervention', manifest)
        self.assertEqual(manifest['intervention'].get('strategy'), 'tiling_overlap')

    def test_mmb_variants_expose_strategy_specific_trainer_mode(self) -> None:
        trainer = resolve_trainer('mmb')
        cases = [
            ('v2_mmb_tiling_overlap', 'tiling_overlap'),
            ('v3_mmb_threshold_nms_calibration', 'threshold_nms_calibration'),
            ('v4_mmb_hard_negative_mining', 'hard_negative_mining'),
        ]
        for slug, strategy in cases:
            manifest = build_run_manifest(slug, 'devsample')
            result = trainer(seed=101, steps=1, manifest=manifest)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['intervention'].get('strategy'), strategy)
            self.assertIn(f'mmb_proxy::{strategy}', result['trainer_mode'])

    def test_mmb_proxy_disables_test_label_conditioning(self) -> None:
        manifest = build_run_manifest('v1_mmb_baseline_reproduction', 'viso')
        trainer = resolve_trainer('mmb')
        result = trainer(seed=101, steps=1, manifest=manifest)
        self.assertEqual(result['status'], 'completed')
        self.assertIn('leakage_controls', result)
        self.assertFalse(result['leakage_controls']['label_conditioning_on_test'])

    def test_v6_calibration_uses_train_split(self) -> None:
        manifest = build_run_manifest('v6_mmb_v5_plus_light_v3_calibration', 'viso')
        trainer = resolve_trainer('mmb')
        result = trainer(seed=101, steps=1, manifest=manifest)
        self.assertEqual(result['status'], 'completed')
        self.assertIsNotNone(result.get('calibration'))
        self.assertEqual(result['calibration']['calibration_split'], 'train')
        self.assertEqual(result['calibration']['evaluation_split'], 'test')

    def test_real_mmb_requires_model_path(self) -> None:
        manifest = build_run_manifest('v1_mmb_baseline_reproduction', 'devsample')
        manifest['intervention'] = {
            'strategy': 'baseline_proxy',
            'inference_mode': 'real',
        }
        trainer = resolve_trainer('mmb')
        result = trainer(seed=101, steps=1, manifest=manifest)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reason'], 'real_inference_failed')
        self.assertIn('model', result.get('error', '').lower())

    def test_complete_mmb_mode_runs(self) -> None:
        manifest = build_run_manifest('v1_mmb_baseline_reproduction', 'devsample')
        manifest['intervention'] = {
            'strategy': 'baseline_proxy',
            'inference_mode': 'complete',
        }
        trainer = resolve_trainer('mmb')
        result = trainer(seed=101, steps=1, manifest=manifest)
        self.assertEqual(result['status'], 'completed')
        self.assertIn('algorithm', result)
        self.assertEqual((result.get('algorithm') or {}).get('name'), 'mmb_complete')


if __name__ == '__main__':
    unittest.main()