import os
from pathlib import Path
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]


def download_dataset():
    """Download the MDP dataset from Roboflow in COCO segmentation format (version 6)."""
    print("Downloading dataset from Roboflow (coco-segmentation format, version 6)...")
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("ziyans-workspace").project("mdp-dataset_accessary")
    version = project.version(2)
    dataset = version.download("coco-mmdetection",
                               location=str(ROOT / "data" / "MDP-Dataset_accessary-2"))
    raw_dir = os.path.abspath(dataset.location)
    print(f"Downloaded to: {raw_dir}")
    return raw_dir


if __name__ == "__main__":
    download_dataset()
