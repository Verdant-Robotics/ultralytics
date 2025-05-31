from ultralytics import YOLO
from training_utils import (
    PrepareDataset,
)
from training_utils import (
    dataset_yaml_path,
    coco_classes_file,
    training_task,
)
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("-l", "--load", type=str, required=True, help="Path to the model weights to load. Load the pretrained model")
    args = parser.parse_args()
    PrepareDataset(coco_classes_file, dataset_yaml_path, training_task)
   
    model = YOLO(args.load)
    model.val()
