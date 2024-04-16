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

    args = parser.parse_args()
    if args.load is not None:
        model = YOLO(args.load)  # Initialize model
    else:
        model = YOLO(GetModelYaml(training_task))  # Initialize model

    PrepareDataset(coco_classes_file, dataset_yaml_path, training_task)

    model.train(
        task=training_task,
        data="verdant.yaml",
        epochs=300,
        flipud=0.5,
        fliplr=0.5,
        scale=0.2,
        mosaic=0.0,  # Please set this to 0.0 TODO: Fix the issue with mosaic and keypoint detection
        imgsz=768,
        seed=1,
        batch=128,
        name=experiment_name,
        device=[0, 1, 2, 3, 4, 5, 6, 7],
    )

    latest_weights_dir = GetLatestWeightsDir()
    Export(f"{latest_weights_dir}/best.pt")  # To export the model to onnx format
