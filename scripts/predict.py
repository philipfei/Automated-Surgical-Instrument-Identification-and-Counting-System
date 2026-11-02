import os
import glob
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    model_path = str(ROOT / "runs" / "segment" / "train" / "weights" / "best.pt")

    if not os.path.exists(model_path):
        print(f"WARNING: trained weights not found at: {model_path}")
        print("Falling back to official pretrained weights for demo only.")
        print("Replace with your best.pt for real inference.")
        model_path = "yolo11m-seg.pt"

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    predict_dir = str(ROOT / "data" / "predict")
    output_dir = os.path.join(predict_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = sorted([
        f for f in glob.glob(os.path.join(predict_dir, "*"))
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not image_files:
        print("No images found in predict/ directory.")
        print(f"Searched: {predict_dir}")
        return

    print(f"Found {len(image_files)} image(s) in predict/")
    print("Running inference (params matched to app/gopro_inference.py)...")

    for img_path in image_files:
        print(f"  {os.path.basename(img_path)}")

    results = model.predict(
        source=image_files,
        imgsz=640,
        conf=0.35,
        iou=0.45,
        device=0,
        save=True,
        save_txt=False,
        project=predict_dir,
        name="output",
        exist_ok=True,
        verbose=False,
    )

    print(f"\nInference complete.")
    print(f"Annotated images saved to: {output_dir}")


if __name__ == "__main__":
    main()
