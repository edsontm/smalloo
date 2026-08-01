"""Motion Modeling Baseline (MMB) package.

Classical motion pipeline for satellite video object detection.
"""

from src.mmb.detection import Detection, MotionDetector, generate_object_proposals
from src.mmb.evaluation import evaluate_detections
from src.mmb.foreground import ForegroundConfig, ForegroundExtractor
from src.mmb.multi_frame_difference import DifferenceResult, MultiFrameDifferencer, MultiFrameDifferenceConfig
from src.mmb.pipeline import MMBPipeline
from src.mmb.robust_matrix_completion import RobustMatrixCompletion, RobustMatrixCompletionConfig

__all__ = [
    'Detection',
    'DifferenceResult',
    'ForegroundConfig',
    'ForegroundExtractor',
    'MMBPipeline',
    'MotionDetector',
    'MultiFrameDifferencer',
    'MultiFrameDifferenceConfig',
    'RobustMatrixCompletion',
    'RobustMatrixCompletionConfig',
    'evaluate_detections',
    'generate_object_proposals',
]
