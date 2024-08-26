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
import torch
from export import Export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("-l", "--load", type=str, default=None, help="Path to the model weights to load. Load the pretrained model")

    args = parser.parse_args()
    model = YOLO(GetModelYaml(training_task), training_task)  # Initialize model
    if args.load is not None:
        if os.path.exists(args.load):
            #model = YOLO(args.load)
            model.load(args.load)

            # Save the model as a workaround. Otherwise distributed training discards the weights.
            tmp_model_name = "/tmp/init.pt"
            torch.save({'model': model.model.state_dict(), 'optimizer': None}, tmp_model_name)
            model = YOLO(tmp_model_name, training_task)
            print("Reloaded model")

        else:
            print(f"[ERROR] : Model {args.load} does not exists")
            exit(1)

    PrepareDataset(coco_classes_file, dataset_yaml_path, training_task)

    model.train(
        task=training_task,
        data="verdant.yaml",
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
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
