from training_utils import (
    GiveModel
)
import os
import argparse


def Export(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print(f"[ERROR] : Model {checkpoint_dir} does not exists")
        exit(1)

    checkpoint_files = ["best.pt", "last.pt"]

    for file in checkpoint_files:
        checkpoint_file_path = os.path.join(checkpoint_dir, file)
        if not os.path.exists(checkpoint_file_path):
            print(f"[ERROR] : Model {checkpoint_file_path} does not exists")
            continue

        prefix = file.split(".")[0]  # "best" or "last"

        model = GiveModel(checkpoint_file_path)

        base_path = checkpoint_dir
        path = model.export(format="onnx", imgsz=[2144, 768], opset=12)
        os.system(f"mv {path} {base_path}/{prefix}_full_height.onnx")

        path = model.export(format="onnx", imgsz=[2144, 4096], opset=12)
        os.system(f"mv {path} {base_path}/{prefix}_full_frame.onnx")

        path = model.export(format="onnx", imgsz=[768, 768], opset=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the model")
    parser.add_argument(
        "-m",
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory where the model weight is. This model will be exported")

    args = parser.parse_args()
    Export(args.checkpoint_dir)
