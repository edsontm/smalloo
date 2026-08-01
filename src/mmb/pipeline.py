from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from src.mmb.detection import Detection, MotionDetector
from src.mmb.evaluation import evaluate_detections
from src.mmb.foreground import ForegroundConfig, ForegroundExtractor
from src.mmb.multi_frame_difference import MultiFrameDifferencer, MultiFrameDifferenceConfig
from src.mmb.robust_matrix_completion import RobustMatrixCompletion, RobustMatrixCompletionConfig
from src.mmb.visualization import draw_detections, save_video


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    detections: List[List[Detection]]
    tracks: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class MMBPipeline:
    """Complete classical Motion Modeling Baseline pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        diff_cfg = config.get('multi_frame_difference', config.get('difference', {}))
        self.differencer = MultiFrameDifferencer(
            MultiFrameDifferenceConfig(
                temporal_windows=tuple(diff_cfg.get('temporal_windows', (1, 2, 3))),
                normalize_frames=bool(diff_cfg.get('normalize_frames', True)),
            )
        )

        completion_cfg = config.get('robust_matrix_completion', config.get('completion', {}))
        self.completion = RobustMatrixCompletion(
            RobustMatrixCompletionConfig(
                lambda_value=completion_cfg.get('lambda_value'),
                mu=completion_cfg.get('mu'),
                rho=float(completion_cfg.get('rho', 1.5)),
                max_iter=int(completion_cfg.get('max_iter', 100)),
                tol=float(completion_cfg.get('tol', 1e-6)),
            )
        )

        fg_cfg = config.get('foreground', {})
        self.foreground = ForegroundExtractor(
            ForegroundConfig(
                threshold=fg_cfg.get('threshold', 'adaptive'),
                kernel_size=int(fg_cfg.get('morphology_kernel', fg_cfg.get('kernel_size', 3))),
                min_component_size=int(fg_cfg.get('min_component_size', 5)),
            )
        )

        det_cfg = config.get('detection', {})
        self.detector = MotionDetector(
            min_area=int(det_cfg.get('min_area', 5)),
            max_area=int(det_cfg.get('max_area', 500)),
        )

    @staticmethod
    def _load_video_frames(video_path: Path) -> List[np.ndarray]:
        try:
            import imageio.v2 as imageio
        except Exception as exc:  # pragma: no cover
            raise RuntimeError('imageio is required for video loading.') from exc

        frames: List[np.ndarray] = []
        reader = imageio.get_reader(str(video_path))
        for frame in reader:
            frames.append(np.asarray(frame))
        reader.close()
        return frames

    @staticmethod
    def _save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _detection_list_to_dict(dets: Sequence[Detection]) -> List[Dict[str, Any]]:
        return [det.to_dict() for det in dets]

    @staticmethod
    def _tracks_to_dict(tracks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(track) for track in tracks]

    @staticmethod
    def _save_array_image(path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if array.ndim == 2:
            norm = array.astype(np.float32)
            max_value = float(norm.max())
            min_value = float(norm.min())
            if max_value > min_value:
                norm = (norm - min_value) / (max_value - min_value)
            else:
                norm = np.zeros_like(norm)
            image = Image.fromarray((norm * 255.0).astype(np.uint8), mode='L')
        else:
            image = Image.fromarray(array.astype(np.uint8))
        image.save(path)

    def run(
        self,
        video_path: Path | None = None,
        output_dir: Path | None = None,
        gt_boxes_by_frame: List[List[List[float]]] | None = None,
        frames: List[np.ndarray] | None = None,
    ) -> PipelineOutput:
        if frames is None:
            if video_path is None:
                raise ValueError('either video_path or frames must be provided')
            frames = self._load_video_frames(video_path)
        if not frames:
            raise ValueError('Could not load frames')

        LOGGER.info('Loaded %d frames', len(frames))

        diff_result = self.differencer.build(frames)
        completion_result = self.completion.decompose(diff_result.differences)
        fg_masks = self.foreground.extract_sequence(completion_result.sparse_frames)

        detections_per_frame: List[List[Detection]] = []
        vis_frames: List[np.ndarray] = []
        for frame, mask in zip(frames, fg_masks):
            detections = self.detector.detect(mask)
            detections_per_frame.append(detections)
            vis_frames.append(draw_detections(frame, detections))

        tracks: List[Dict[str, Any]] = []
        vis_track_frames = list(vis_frames)

        if output_dir is None:
            output_dir = Path('results/mmb')
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_json(
            output_dir / 'detections.json',
            [self._detection_list_to_dict(d) for d in detections_per_frame],
        )
        self._save_json(output_dir / 'tracks.json', self._tracks_to_dict(tracks))

        diff_dir = output_dir / 'temporal_differences'
        low_rank_dir = output_dir / 'low_rank_frames'
        sparse_dir = output_dir / 'sparse_frames'
        fg_dir = output_dir / 'foreground_masks'
        diff_dir.mkdir(parents=True, exist_ok=True)
        low_rank_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        fg_dir.mkdir(parents=True, exist_ok=True)
        for idx, diff in enumerate(diff_result.differences):
            np.save(diff_dir / f'{idx:05d}.npy', diff)
        for idx, low_rank in enumerate(completion_result.low_rank_frames):
            np.save(low_rank_dir / f'{idx:05d}.npy', low_rank)
        for idx, sparse in enumerate(completion_result.sparse_frames):
            np.save(sparse_dir / f'{idx:05d}.npy', sparse)
        for idx, mask in enumerate(fg_masks):
            np.save(fg_dir / f'{idx:05d}.npy', mask)

        if completion_result.low_rank_frames:
            self._save_array_image(output_dir / 'background.png', completion_result.low_rank_frames[0])

        save_video(vis_track_frames, output_dir / 'visualization.mp4')

        det_metrics = {'precision': 0.0, 'recall': 0.0, 'ap': 0.0, 'f1': 0.0, 'false_alarms': 0.0, 'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
        if gt_boxes_by_frame is not None and len(gt_boxes_by_frame) == len(detections_per_frame):
            evaluation = evaluate_detections(detections_per_frame, gt_boxes_by_frame)
            det_metrics = {
                'precision': evaluation.precision,
                'recall': evaluation.recall,
                'ap': evaluation.ap,
                'f1': (2.0 * evaluation.precision * evaluation.recall / (evaluation.precision + evaluation.recall))
                if (evaluation.precision + evaluation.recall) > 0
                else 0.0,
                'false_alarms': evaluation.false_alarms,
                'tp': evaluation.tp,
                'fp': evaluation.fp,
                'fn': evaluation.fn,
            }

        trk_metrics = {'num_tracks': 0.0, 'mean_track_length': 0.0, 'mota': 0.0, 'motp': 0.0, 'idf1': 0.0, 'id_switches': 0.0}

        metrics_payload = {
            'detection': det_metrics,
            'tracking': trk_metrics,
            'num_frames': len(frames),
            'temporal_windows': diff_result.windows,
            'robust_matrix_completion': {
                'iterations': completion_result.iterations,
                'residual_norm': completion_result.residual_norm,
            },
        }
        self._save_json(output_dir / 'metrics.json', metrics_payload)

        return PipelineOutput(detections=detections_per_frame, tracks=tracks, metrics=metrics_payload)
