from ultralytics import YOLO
from training_utils import (
    PrepareDataset,
    GetModelYaml,
    GetLatestWeightsDir,
)
from training_utils import (
    dataset_yaml_path,
    coco_classes_file,
    training_task,
    experiment_name,
)
import os
import argparse
from export import Export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("-l", "--load", type=str, default=None, help="Path to the model weights to load. Load the pretrained model")
    parser.add_argument("-d", "--disable-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    PrepareDataset(coco_classes_file, dataset_yaml_path, training_task)

    # args.load = 'runs/neptune/pose-segmentation/train27/weights/best.pt'

    args.load = 'runs/pose-segmentation/train12/weights/best.pt'
    
    model = YOLO(args.load)

    model.val()
