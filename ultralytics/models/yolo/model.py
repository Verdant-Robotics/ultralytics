# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models import yolo  # noqa
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, PoseModel, PoseContrastiveModel, 
                                  PoseMultiClsHeadsModel, PoseTunableHeadModel, SegmentationModel)


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, }, 
            'pose-contrastive': {
                'model': PoseContrastiveModel,
                'trainer': yolo.pose.PoseContrastiveTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, },
            'pose-multiclsheads': {
                'model': PoseMultiClsHeadsModel,
                'trainer': yolo.pose.PoseMultiClsHeadsTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, },
            'pose-tunablehead': {
                'model': PoseTunableHeadModel,
                'trainer': yolo.pose.PoseTunableHeadTrainer,
                'validator': yolo.pose.PoseTunableHeadValidator,
                'predictor': yolo.pose.PosePredictor, },
            }
