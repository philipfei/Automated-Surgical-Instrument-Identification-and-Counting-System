import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    yaml_path = os.path.abspath("C:/YOLO_Data/yolo_surgical_dataset/dataset.yaml")
    
    if not os.path.exists(yaml_path):
        yaml_path = str(ROOT / "data" / "yolo_surgical_dataset" / "dataset.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset configuration file not found at: {yaml_path}")

    print(f"Successfully loaded dataset configuration: {yaml_path}")
    print("Loading pre-trained model weights...")
    
    model = YOLO("yolo11m-seg.pt")

    print("Starting training pipeline on NVIDIA A4500...")
    model.train(
        data=yaml_path,
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=16,
        cache=True,
        plots=True,
        project=str(ROOT / "runs" / "segment"),
        # ── Aggressive augmentation for domain generalization ──
        degrees=30.0,
        perspective=0.0005,
        shear=2.0,
        scale=0.7,
        translate=0.15,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.03,
        hsv_s=0.8,
        hsv_v=0.5,
        mosaic=1.0,
        close_mosaic=20,
        mixup=0.15,
        cutmix=0.0,
        copy_paste=0.3,
        copy_paste_mode="flip",
        erasing=0.2,
    )
    print("\nTraining completed successfully!")
    print("Best weights saved at: ./runs/segment/train/weights/best.pt")

    # Evaluate on the test split with full metric plots
    print("\n" + "=" * 60)
    print("  Evaluating best checkpoint on TEST split...")
    print("=" * 60)

    best_path = str(ROOT / "runs" / "segment" / "train" / "weights" / "best.pt")
    model = YOLO(best_path)
    metrics = model.val(
        data=yaml_path,
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        save_json=False,
        project=str(ROOT / "runs" / "segment"),
    )

    print("\n" + "=" * 60)
    print("  TEST SET RESULTS")
    print("=" * 60)
    print(f"  Box mAP50:     {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Mask mAP50:    {metrics.seg.map50:.4f}")
    print(f"  Mask mAP50-95: {metrics.seg.map:.4f}")
    print(f"\n  Per-class metrics and plots saved to: ./runs/segment/val/")
    print("=" * 60)

if __name__ == "__main__":
    main()