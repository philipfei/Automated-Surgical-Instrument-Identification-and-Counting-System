import requests
import cv2
import numpy as np
import time
import os
import threading
from pathlib import Path
from ultralytics import YOLO

GOPRO_IP = "172.23.111.51"
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(ROOT / "runs" / "segment" / "train" / "weights" / "best.pt")
DISPLAY_W, DISPLAY_H = 1280, 720

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "fflags;nobuffer+discardcorrupt"
    "|flags;low_delay"
    "|f;mpegts"
    "|probesize;5000000"
    "|analyzeduration;1000000"
    "|reorder_queue_size;0"
    "|max_delay;0"
)

COLORS = np.random.default_rng(42).integers(50, 220, size=(80, 3), dtype=np.uint8)


def _draw_segmentation(frame, results):
    overlay = frame.copy()
    r = results[0]

    if r.masks is None:
        return frame

    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    names = r.names
    h, w = frame.shape[:2]

    for i, (mask, cls_id, conf) in enumerate(zip(masks, class_ids, confs)):
        color = COLORS[cls_id % len(COLORS)].tolist()
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = mask_resized > 0.5
        overlay[binary] = (
            np.array(overlay[binary], dtype=np.float32) * 0.45
            + np.array(color, dtype=np.float32) * 0.55
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        if boxes.xyxy is not None:
            x1, y1 = int(boxes.xyxy[i][0]), int(boxes.xyxy[i][1])
            label = f"{names[cls_id]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _draw_counts(frame, results):
    """Draw instrument count panel on the top-right of the frame."""
    r = results[0]
    if r.masks is None:
        return

    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    name_map = r.names
    counts = {}
    for cid in class_ids:
        name = name_map.get(cid, f"cls_{cid}")
        counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())
    line_h = 21
    x0 = DISPLAY_W - 250
    y0 = 36
    n_lines = len(counts) + 1

    panel_h = n_lines * line_h + 14
    x1, y1 = x0 - 10, y0 - 26
    x2, y2 = x0 + 242, y0 - 26 + panel_h
    panel_roi = frame[y1:y2, x1:x2].copy()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(panel_roi, 0.35, frame[y1:y2, x1:x2], 0.65, 0,
                    frame[y1:y2, x1:x2])

    cv2.putText(frame, f"Instruments: {total}", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
    y = y0 + line_h
    for name in sorted(counts):
        cv2.putText(frame, f"{name}: {counts[name]}", (x0 + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += line_h


# ── GoPro helpers ───────────────────────────────────────────

def _start_gopro(resolution="720"):
    try:
        r = requests.get(
            f"http://{GOPRO_IP}/gp/gpWebcam/START?res={resolution}",
            timeout=5)
        print(f"GoPro START: {r.status_code}")
        time.sleep(2)
        return True
    except Exception as e:
        print(f"GoPro start failed: {e}")
        return False


def _stop_gopro():
    try:
        requests.get(f"http://{GOPRO_IP}/gp/gpWebcam/STOP", timeout=3)
        print("GoPro stream stopped")
    except Exception:
        pass


def _open_capture():
    cap = cv2.VideoCapture("udp://0.0.0.0:8554", cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ── Inference class ─────────────────────────────────────────

class GoProInference:
    """Manages GoPro capture + YOLO inference in a background thread.

    Usage:
        infer = GoProInference()
        infer.start()
        frame = infer.get_frame()      # latest annotated frame (BGR numpy)
        counts = infer.get_counts()    # {class_name: count}
        infer.stop()
    """

    def __init__(self, model_path=None, conf=0.55, iou=0.5, imgsz=640):
        self.model_path = model_path or MODEL_PATH
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_vis = None
        self._latest_counts = {}
        self._stop_event = threading.Event()
        self._cap_ref = None
        self._model = None
        self._running = False

    # ── public API ──

    @property
    def is_running(self):
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True

        print("Loading YOLO model...")
        self._model = YOLO(self.model_path)
        dummy = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        print("Model ready")

        print("Starting GoPro webcam (720p)...")
        _start_gopro("720")
        cap = _open_capture()
        if not cap.isOpened():
            _stop_gopro()
            raise RuntimeError("Failed to open GoPro video stream")
        self._cap_ref = [cap]

        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()
        print("GoPro inference running")

    def stop(self):
        self._running = False
        self._stop_event.set()

        if self._cap_ref and self._cap_ref[0]:
            self._cap_ref[0].release()
        _stop_gopro()

        with self._lock:
            self._latest_frame = None
            self._latest_vis = None
            self._latest_counts = {}
        print("GoPro inference stopped")

    def get_frame(self):
        """Return the latest annotated frame as a BGR numpy array, or None."""
        with self._lock:
            return self._latest_vis.copy() if self._latest_vis is not None else None

    def get_counts(self):
        """Return dict {class_name: count} from the latest frame, or {}."""
        with self._lock:
            return dict(self._latest_counts)

    # ── internal threads ──

    def _capture_loop(self):
        fail_count = 0
        max_fails = 30
        while not self._stop_event.is_set():
            cap = self._cap_ref[0]
            ret, frame = cap.read()
            if not ret or frame is None:
                fail_count += 1
                if fail_count >= max_fails:
                    print("Stream lost, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    _start_gopro("720")
                    self._cap_ref[0] = _open_capture()
                    fail_count = 0
                continue
            fail_count = 0
            with self._lock:
                self._latest_frame = frame

    def _inference_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None  # consume once

            if frame is None:
                time.sleep(0.005)
                continue

            results = self._model(
                frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                half=False, device=0, verbose=False, stream=False)

            vis = _draw_segmentation(frame, results)

            # FPS / HUD left side
            cv2.putText(vis, f"Res: {DISPLAY_W}x{DISPLAY_H}", (12, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            vis_display = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
            _draw_counts(vis_display, results)

            # Extract counts
            r = results[0]
            counts = {}
            if r.masks is not None:
                cids = r.boxes.cls.cpu().numpy().astype(int)
                names = r.names
                for cid in cids:
                    name = names.get(cid, f"cls_{cid}")
                    counts[name] = counts.get(name, 0) + 1

            with self._lock:
                self._latest_vis = vis_display
                self._latest_counts = counts


# ── Standalone mode (original cv2.imshow behaviour) ─────────

def standalone():
    """Original behaviour: run inference with its own OpenCV window."""
    infer = GoProInference()
    infer.start()

    cv2.namedWindow("GoPro + YOLO Seg", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GoPro + YOLO Seg", DISPLAY_W, DISPLAY_H)
    print("Keys: q = quit | f = fullscreen")

    try:
        while True:
            frame = infer.get_frame()
            if frame is not None:
                cv2.imshow("GoPro + YOLO Seg", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                prop = cv2.getWindowProperty("GoPro + YOLO Seg", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(
                    "GoPro + YOLO Seg", cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if prop != cv2.WINDOW_FULLSCREEN
                    else cv2.WINDOW_NORMAL)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        infer.stop()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    standalone()
