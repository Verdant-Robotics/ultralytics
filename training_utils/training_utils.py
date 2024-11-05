import os
from ultralytics import YOLO


dataset_yaml_path = os.environ.get("DATASET_YAML", 
                                   "/training/ultralytics/ultralytics/cfg/datasets/verdant.yaml")
coco_classes_file = os.environ.get("COCO_LABELS", "/dataset/classes.txt")
runs_directory = os.environ.get("RUNS_DIR", "/training/runs")
training_task = os.environ.get("TASK", "detect")   # 'detect' for bbox | 'pose' for hybrid model
experiment_name = os.environ.get("EXP_NAME", None)  # Results will be saved with this name  under runs/<task>/<exp_name>


def PrepareDataset(coco_classes_file, dataset_yaml, training_task):
    classes = []
    with open(coco_classes_file, "r") as f:
        for l in f.readlines():
            l = l.strip("\n")
            l = l.strip(" ")
            classes.append(l)

    with open(dataset_yaml, "w") as f:
        f.write("path: /dataset\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        for i in range(len(classes)):
            f.write(f"  {i}: {classes[i]}\n")
        if training_task == "pose" or training_task == "pose-contrastive":
            f.write("\nkpt_shape: [1, 3]\n")  # enforce keypoint shape to [1, 3] for pose models
    return


def GetModelYaml(task):
    if task == "detect":
        return "yolov8n.yaml"
    elif task == "pose":
        return "yolov8n-pose.yaml"
    elif task == "pose-contrastive":
        return "yolov8n-pose-contrastive.yaml"
    print(f"Unknown task {task}")
    return None


def GiveModel(ckpt_path):
    if os.path.exists(ckpt_path):
        return YOLO(ckpt_path)
    print(f"[ERROR] : Model {ckpt_path} does not exists")
    return None


def GetLatestWeightsDir():
    base_dir = f"{runs_directory}/{training_task}"

    # Get all entries in the directory given by path
    entries = os.listdir(base_dir)
    # Filter entries to only include directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(base_dir, entry))]
    # Sort directories by creation time
    sorted_directories = sorted(
        directories,
        key=lambda x: os.path.getctime(os.path.join(base_dir, x)))

    if len(sorted_directories) == 0:
        print(f"[ERROR] : No weights directory found in {base_dir}")
        return None
    return f"{base_dir}/{sorted_directories[-1]}/weights"


def LoadBestModel():
    best_ckpt = f"{GetLatestWeightsDir()}/best.pt"
    return GiveModel(best_ckpt)
