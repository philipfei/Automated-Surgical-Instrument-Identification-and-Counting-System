import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    checkpoint_path = str(ROOT / "runs" / "segment" / "train" / "weights" / "last.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No interrupted training checkpoint found at: {checkpoint_path}")

    print(f"Found existing training checkpoint: {checkpoint_path}")
    print("Resuming training pipeline from last saved state...")
    
    model = YOLO(checkpoint_path)
    
    model.train(resume=True)
    
    print("\nResumed training completed successfully!")

if __name__ == "__main__":
    main()