"""
MediScan — Surgical Instrument Detection GUI
Camera feed from gopro_inference, Arduino control for tray shaking + weighing.

Requirements: pip install pillow pyserial
Run with:     python app/ui.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import sys
from collections import Counter
from PIL import Image, ImageTk
import cv2
import numpy as np

# ── Serial (pyserial) ─────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("WARNING: pyserial not installed — Arduino communication disabled.")
    print("  Install with: pip install pyserial")

# ── GoPro inference ───────────────────────────────────────
try:
    from gopro_inference import GoProInference
    HAS_GOPRO = True
except ImportError:
    HAS_GOPRO = False
    print("WARNING: gopro_inference module not found — camera feed disabled.")

# ── Colour palette ────────────────────────────────────────
BG_DARK      = "#0A0E1A"
BG_PANEL     = "#0F1528"
BG_CARD      = "#141B30"
ACCENT       = "#00C9A7"
ACCENT2      = "#0084FF"
WARN         = "#FFB830"
DANGER       = "#FF4757"
TEXT_PRIMARY = "#E8EDF5"
TEXT_DIM     = "#5A6A8A"
BORDER       = "#1E2A44"

# ── Detection config ─────────────────────────────────────
SHAKE_DURATION_S = 5.0                # motor shake duration (seconds)
WEIGHT_MISSING_THRESHOLD_G = 20.0    # weight drop threshold (grams) to flag missing

# ── 4 detection classes (in dataset order) ────────────────
CLASS_NAMES = ["Thumb_Forceps", "Forceps", "Scissors", "Nasal_Speculum"]


# ── Serial helper ─────────────────────────────────────────

def list_serial_ports():
    """Return list of available serial port device names."""
    if not HAS_SERIAL:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


class ArduinoLink:
    """Background thread for Arduino serial communication."""

    def __init__(self, baud=115200):
        self.port = None
        self.baud = baud
        self._ser = None
        self._lock = threading.Lock()
        self._weight = 0.0
        self._running = False
        self._thread = None
        self.connected = False

    def connect(self, port):
        """Connect to the given serial port."""
        if not HAS_SERIAL or not port:
            return False
        # Disconnect any existing connection first
        self.disconnect()
        try:
            self.port = port
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
            time.sleep(1.5)  # let Arduino reset
            self.connected = True
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"Arduino connected on {self.port}")
            return True
        except Exception as e:
            print(f"Arduino connect failed: {e}")
            self.port = None
            return False

    def disconnect(self):
        self._running = False
        if self._ser and self._ser.is_open:
            self._ser.close()
        self.connected = False

    def _read_loop(self):
        buf = ""
        while self._running:
            try:
                if self._ser.in_waiting:
                    data = self._ser.read(self._ser.in_waiting).decode("utf-8", errors="replace")
                    buf += data
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line.startswith("Filtered:"):
                            try:
                                w = float(line.split(":")[1].strip())
                                with self._lock:
                                    self._weight = w
                            except ValueError:
                                pass
            except Exception:
                time.sleep(0.1)

    def get_weight(self):
        with self._lock:
            return self._weight

    def reset_weight(self):
        """Force weight to zero (call after Arduino RESET)."""
        with self._lock:
            self._weight = 0.0

    def send(self, cmd):
        """Send a command and optionally wait for the OK response."""
        if self._ser and self._ser.is_open:
            self._ser.write((cmd + "\n").encode("utf-8"))
            self._ser.flush()
            time.sleep(0.2)  # wait for Arduino to process
            # Read response (Arduino sends "OK:CMD")
            # But don't block too long
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode("utf-8", errors="replace").strip()
                    if line.startswith("OK:"):
                        return line
                time.sleep(0.05)
            return None


# ── Main Application ──────────────────────────────────────

class MediScanApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("MediScan — Instrument Detection System  v3.0")
        self.geometry("1100x720")
        self.minsize(900, 620)
        self.configure(bg=BG_DARK)

        # ── State ──
        self.infer = None
        self.arduino = None
        self._camera_active = True
        self._device_connected = True
        self._baseline_counts = None      # set after initial detection
        self._final_counts = None         # set after final detection
        self._initial_weight = 0.0        # weight recorded at initial detection
        self._final_weight = 0.0          # weight recorded at final detection
        self._detecting = False
        self._det_start_time = 0
        self._det_status_text = tk.StringVar(value="Idle — press Initial or Final Detection")
        self._det_progress = tk.DoubleVar(value=0.0)

        # Instrument checklist state
        self._inst_counts = {name: 0 for name in CLASS_NAMES}
        self._inst_status = {name: " " for name in CLASS_NAMES}   # "present" / "MISSING" / " "
        self._inst_conf = {name: " " for name in CLASS_NAMES}

        self._build_ui()
        self._start_subsystems()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Subsystem startup ──────────────────────────────

    def _start_subsystems(self):
        # GoPro inference
        if HAS_GOPRO:
            try:
                self.infer = GoProInference()
                self.infer.start()
                self.camera_card.configure(text="ACTIVE", fg=ACCENT)
                self.cam_dot.configure(fg=ACCENT)
            except Exception as e:
                print(f"Camera start failed: {e}")
                self.camera_card.configure(text="OFFLINE", fg=DANGER)
                self.cam_dot.configure(fg=DANGER)
        else:
            self.camera_card.configure(text="NO MODULE", fg=DANGER)

        # Arduino (manual connect via COM port selector)
        self.arduino = ArduinoLink()
        self.device_card.configure(text="OFFLINE", fg=DANGER)

        # Camera feed poll
        self._update_feed()

        # Weight poll
        self._update_weight()

    # ── UI construction ────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self, bg=BG_PANEL, height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="MEDI", bg=BG_PANEL, fg=ACCENT,
                 font=("Courier", 20, "bold")).pack(side="left", padx=(18, 0), pady=10)
        tk.Label(header, text="SCAN", bg=BG_PANEL, fg=TEXT_PRIMARY,
                 font=("Courier", 20, "bold")).pack(side="left", pady=10)
        tk.Label(header, text=" · Smart Surgical Instrument Detection",
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Courier", 10)).pack(side="left", pady=14)

        # ── Body ──
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        # Left: camera feed
        self.feed_frame = tk.Frame(body, bg=BG_CARD,
                                   highlightthickness=1, highlightbackground=BORDER)
        self.feed_frame.pack(side="left", fill="both", expand=True)
        self.feed_frame.pack_propagate(False)

        feed_hdr = tk.Frame(self.feed_frame, bg=BG_PANEL, height=32)
        feed_hdr.pack(fill="x")
        feed_hdr.pack_propagate(False)
        self.cam_dot = tk.Label(feed_hdr, text="●", fg=TEXT_DIM,
                                bg=BG_PANEL, font=("Courier", 10))
        self.cam_dot.pack(side="left", padx=(12, 4), pady=6)
        tk.Label(feed_hdr, text="CAMERA FEED  —  OR-CAM-02",
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(side="left", pady=6)

        self.cam_label = tk.Label(self.feed_frame, bg=BG_CARD, text="No feed", fg=TEXT_DIM)
        self.cam_label.pack(fill="both", expand=True)

        # Right panel
        panel = tk.Frame(body, bg=BG_DARK, width=280)
        panel.pack(side="right", fill="y", padx=(10, 0))
        panel.pack_propagate(False)

        # ── Status cards ──
        self._section(panel, "SYSTEM STATUS")
        self.device_card = self._status_card(panel, "DEVICE", "ONLINE", ACCENT)
        self.camera_card = self._status_card(panel, "CAMERA", "ACTIVE", ACCENT)
        self.missing_card = self._status_card(panel, "MISSING ITEMS", "0", ACCENT)
        self.weight_card = self._status_card(panel, "WEIGHT (g)", "0.0", ACCENT)

        # ── COM port selector ──
        com_row = tk.Frame(panel, bg=BG_DARK)
        com_row.pack(fill="x", pady=(4, 0))
        tk.Label(com_row, text="COM PORT", bg=BG_DARK, fg=TEXT_DIM,
                 font=("Courier", 8)).pack(side="left")

        self._combo_var = tk.StringVar()
        self._combo = ttk.Combobox(com_row, textvariable=self._combo_var,
                                    values=list_serial_ports(),
                                    state="readonly", width=10)
        self._combo.pack(side="left", padx=(4, 2))
        self._combo.bind("<<ComboboxSelected>>", lambda e: None)

        self._com_btn = tk.Button(
            com_row, text="CONNECT", bg=ACCENT2, fg="white",
            font=("Courier", 7, "bold"), relief="flat", cursor="hand2",
            padx=6, pady=2, command=self._on_com_toggle)
        self._com_btn.pack(side="left", padx=2)

        refresh_btn = tk.Button(
            com_row, text="↻", bg=BG_CARD, fg=TEXT_DIM,
            font=("Courier", 8), relief="flat", cursor="hand2",
            padx=4, pady=2, command=self._on_refresh_ports)
        refresh_btn.pack(side="left", padx=2)

        # ── Buttons ──
        self._section(panel, "DETECTION CONTROLS")

        btn_frame = tk.Frame(panel, bg=BG_DARK)
        btn_frame.pack(fill="x", pady=(0, 4))

        self.reset_btn = tk.Button(
            btn_frame, text="⟳  RESET", bg=ACCENT2, fg=TEXT_PRIMARY,
            font=("Courier", 10, "bold"), relief="flat", cursor="hand2",
            activebackground="#0066cc", activeforeground="white",
            padx=0, pady=8, command=self._on_reset)
        self.reset_btn.pack(fill="x", pady=2)

        self.estop_btn = tk.Button(
            btn_frame, text="■  EMERGENCY STOP", bg=DANGER, fg="white",
            font=("Courier", 10, "bold"), relief="flat", cursor="hand2",
            activebackground="#cc3344", activeforeground="white",
            padx=0, pady=8, command=self._on_emergency_stop)
        self.estop_btn.pack(fill="x", pady=2)

        self.initial_btn = tk.Button(
            btn_frame, text="⬤  INITIAL DETECTION", bg=ACCENT, fg=BG_DARK,
            font=("Courier", 10, "bold"), relief="flat", cursor="hand2",
            activebackground="#00a88c", activeforeground=BG_DARK,
            padx=0, pady=8, command=self._on_initial_detection)
        self.initial_btn.pack(fill="x", pady=2)

        self.final_btn = tk.Button(
            btn_frame, text="■  FINAL DETECTION", bg="#3A3F5C", fg=TEXT_PRIMARY,
            font=("Courier", 10, "bold"), relief="flat", cursor="hand2",
            activebackground="#4A4F6C", activeforeground="white",
            padx=0, pady=8, command=self._on_final_detection,
            state="disabled")
        self.final_btn.pack(fill="x", pady=2)

        # ── Detection status ──
        det_box = tk.Frame(panel, bg=BG_CARD,
                           highlightthickness=1, highlightbackground=BORDER)
        det_box.pack(fill="x", pady=(6, 8))
        tk.Label(det_box, text="DETECTION STATUS", bg=BG_CARD, fg=TEXT_DIM,
                 font=("Courier", 8)).pack(anchor="w", padx=8, pady=(6, 0))
        tk.Label(det_box, textvariable=self._det_status_text,
                 bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=("Courier", 10, "bold"),
                 wraplength=240, justify="left").pack(anchor="w", padx=8, pady=(2, 4))

        self.det_prog_bar = ttk.Progressbar(
            det_box, variable=self._det_progress,
            maximum=100, mode="determinate",
            style="Teal.Horizontal.TProgressbar")
        self.det_prog_bar.pack(fill="x", padx=8, pady=(0, 8))

        # ── Instrument checklist ──
        self._section(panel, "INSTRUMENT CHECKLIST")

        list_outer = tk.Frame(panel, bg=BG_CARD,
                              highlightthickness=1, highlightbackground=BORDER)
        list_outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(list_outer, bg=BG_CARD, highlightthickness=0)
        scroll = ttk.Scrollbar(list_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._list_inner = tk.Frame(canvas, bg=BG_CARD)
        canvas.create_window((0, 0), window=self._list_inner, anchor="nw")
        self._list_inner.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self._checklist_widgets = {}  # name -> {dot, status_lbl}
        self._rebuild_checklist()

        # ── Hardware toggles ──
        self._section(panel, "HARDWARE TOGGLES")
        tog_row = tk.Frame(panel, bg=BG_DARK)
        tog_row.pack(fill="x", pady=(0, 4))

        self._dev_tog_var = tk.BooleanVar(value=True)
        self._cam_tog_var = tk.BooleanVar(value=True)
        self._toggle_btn(tog_row, "DEVICE", self._dev_tog_var, self._on_device_toggle)
        self._toggle_btn(tog_row, "CAMERA", self._cam_tog_var, self._on_camera_toggle)

        # Progress bar style
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Teal.Horizontal.TProgressbar",
                        troughcolor=BG_PANEL, background=ACCENT,
                        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)

    # ── Checklist ───────────────────────────────────────

    def _rebuild_checklist(self):
        for w in self._list_inner.winfo_children():
            w.destroy()
        self._checklist_widgets.clear()

        for name in CLASS_NAMES:
            row = tk.Frame(self._list_inner, bg=BG_CARD)
            row.pack(fill="x", padx=6, pady=2)

            status = self._inst_status.get(name, " ")
            count = self._inst_counts.get(name, 0)

            if status == "MISSING":
                dot_col, text = DANGER, f"MISSING ({count})"
            elif status == "WEIGHT_MISSING":
                dot_col, text = DANGER, "MISSING"
            elif status == "present":
                dot_col, text = ACCENT, str(count)
            else:
                dot_col, text = TEXT_DIM, "-"

            dot = tk.Label(row, text="●", fg=dot_col, bg=BG_CARD,
                           font=("Courier", 9))
            dot.pack(side="left", padx=(2, 4))

            tk.Label(row, text=name, bg=BG_CARD, fg=TEXT_PRIMARY,
                     font=("Courier", 9), anchor="w").pack(
                side="left", fill="x", expand=True)

            status_lbl = tk.Label(row, text=text, bg=BG_CARD,
                                  fg=dot_col if status != " " else TEXT_DIM,
                                  font=("Courier", 9, "bold"))
            status_lbl.pack(side="right", padx=4)

            self._checklist_widgets[name] = {"dot": dot, "status_lbl": status_lbl}

    def _update_checklist_display(self):
        """Update existing widget text/colors without rebuilding."""
        for name in CLASS_NAMES:
            w = self._checklist_widgets.get(name)
            if not w:
                continue
            status = self._inst_status.get(name, " ")
            count = self._inst_counts.get(name, 0)

            if status == "MISSING":
                dot_col, text = DANGER, f"MISSING ({count})"
            elif status == "WEIGHT_MISSING":
                dot_col, text = DANGER, "MISSING"
            elif status == "present":
                dot_col, text = ACCENT, str(count)
            else:
                dot_col, text = TEXT_DIM, "-"

            w["dot"].configure(fg=dot_col)
            w["status_lbl"].configure(text=text, fg=dot_col if status != " " else TEXT_DIM)

    # ── Widget helpers ──────────────────────────────────

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=BG_DARK)
        f.pack(fill="x", pady=(10, 4))
        tk.Label(f, text=title, bg=BG_DARK, fg=TEXT_DIM,
                 font=("Courier", 8)).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=8)

    def _status_card(self, parent, label, value, color):
        card = tk.Frame(parent, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x", pady=2)
        inner = tk.Frame(card, bg=BG_CARD)
        inner.pack(fill="x", padx=10, pady=6)
        tk.Label(inner, text=label, bg=BG_CARD, fg=TEXT_DIM,
                 font=("Courier", 8)).pack(side="left")
        val_lbl = tk.Label(inner, text=value, bg=BG_CARD, fg=color,
                           font=("Courier", 9, "bold"))
        val_lbl.pack(side="right")
        return val_lbl

    def _toggle_btn(self, parent, label, var, cmd):
        f = tk.Frame(parent, bg=BG_DARK)
        f.pack(side="left", expand=True, fill="x", padx=2)
        tk.Label(f, text=label, bg=BG_DARK, fg=TEXT_DIM,
                 font=("Courier", 8)).pack()
        cb = tk.Checkbutton(f, variable=var, bg=BG_DARK,
                            activebackground=BG_DARK, selectcolor=BG_DARK,
                            fg=ACCENT, activeforeground=ACCENT,
                            indicatoron=False, text="ON",
                            font=("Courier", 9, "bold"),
                            width=7, pady=4, relief="flat", cursor="hand2",
                            command=cmd)
        cb.pack()
        return cb

    # ── Camera feed polling ─────────────────────────────

    def _update_feed(self):
        if self.infer and self.infer.is_running:
            frame = self.infer.get_frame()
            if frame is not None:
                # BGR -> RGB -> PIL -> ImageTk
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to fit the panel while keeping aspect ratio
                panel_w = max(100, self.feed_frame.winfo_width() - 4)
                panel_h = max(100, self.feed_frame.winfo_height() - 36)
                h, w = rgb.shape[:2]
                scale = min(panel_w / w, panel_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w > 0 and new_h > 0:
                    rgb = cv2.resize(rgb, (new_w, new_h))
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.cam_label.configure(image=imgtk, text="")
                self.cam_label.image = imgtk  # keep ref

        self.after(30, self._update_feed)

    # ── Weight polling ──────────────────────────────────

    def _update_weight(self):
        if self.arduino and self.arduino.connected:
            w = self.arduino.get_weight()
            self.weight_card.configure(text=f"{w:.1f}")
        self.after(200, self._update_weight)

    # ── Button handlers ─────────────────────────────────

    def _on_reset(self):
        if self._detecting:
            return
        if self.arduino and self.arduino.connected:
            threading.Thread(target=self._arduino_reset, daemon=True).start()

    def _arduino_reset(self):
        self.after(0, lambda: self._det_status_text.set("Stopping motor + resetting..."))
        # 1) Stop motor
        if self.arduino and self.arduino.connected:
            self.arduino.send("STOP")
        # 2) Reset weighing system
        if self.arduino and self.arduino.connected:
            self.arduino.send("RESET")
        self.arduino.reset_weight()
        # 3) Clear detection state
        self._baseline_counts = None
        self._final_counts = None
        self._initial_weight = 0.0
        self._final_weight = 0.0
        for name in CLASS_NAMES:
            self._inst_counts[name] = 0
            self._inst_status[name] = " "
        self.after(0, self._update_checklist_display)
        self.after(0, lambda: self.final_btn.configure(
            state="disabled", bg="#3A3F5C", fg=TEXT_PRIMARY))
        self.after(0, lambda: self.missing_card.configure(text="0", fg=ACCENT))
        self.after(0, lambda: self.weight_card.configure(text="0.0"))
        self.after(0, lambda: self._det_status_text.set(
            "Reset complete — place instruments"))

    def _on_emergency_stop(self):
        """Immediately stop the motor (always enabled)."""
        if self.arduino and self.arduino.connected:
            threading.Thread(target=lambda: (
                self.arduino.send("STOP"),
                self.after(0, lambda: self._det_status_text.set("EMERGENCY STOP — motor halted"))
            ), daemon=True).start()

    def _on_com_toggle(self):
        """Connect or disconnect the selected COM port."""
        if self.arduino and self.arduino.connected:
            # Disconnect
            self.arduino.disconnect()
            self.device_card.configure(text="OFFLINE", fg=DANGER)
            self._com_btn.configure(text="CONNECT", bg=ACCENT2)
        else:
            # Connect
            port = self._combo_var.get()
            if not port:
                self._det_status_text.set("Select a COM port first")
                return
            if self.arduino.connect(port):
                self.device_card.configure(text="ONLINE", fg=ACCENT)
                self._com_btn.configure(text="DISCONNECT", bg=DANGER)
            else:
                self._det_status_text.set(f"Failed to connect to {port}")

    def _on_refresh_ports(self):
        """Refresh the COM port list."""
        self._combo["values"] = list_serial_ports()
        if list_serial_ports():
            self._combo_var.set(list_serial_ports()[0])

    def _on_initial_detection(self):
        if self._detecting:
            return
        self._detecting = True
        self._set_buttons_state("disabled")
        self._det_start_time = time.time()
        threading.Thread(target=self._run_detection, args=("initial",), daemon=True).start()

    def _on_final_detection(self):
        if self._detecting:
            return
        if self._baseline_counts is None:
            self._det_status_text.set("Run Initial Detection first")
            return
        self._detecting = True
        self._set_buttons_state("disabled")
        self._det_start_time = time.time()
        threading.Thread(target=self._run_detection, args=("final",), daemon=True).start()

    def _set_buttons_state(self, state):
        self.reset_btn.configure(state=state)
        self.initial_btn.configure(state=state)
        self.final_btn.configure(state=state)

    # ── Detection workflow ──────────────────────────────

    def _run_detection(self, mode):
        """Run the full detection sequence in a background thread.

        Workflow: weigh → wait 1s → (shake → detect) × 5 → take mode.
        """
        try:
            # 0) Record weight, then pause 1s before shaking
            current_weight = self.arduino.get_weight() if self.arduino else 0.0
            if mode == "initial":
                self._initial_weight = current_weight
            else:
                self._final_weight = current_weight
            self.after(0, lambda: self._det_progress.set(0))
            self._update_status_threadsafe(
                f"Weight recorded: {current_weight:.1f}g — waiting 1s...")
            time.sleep(1.0)

            # 1) Shake + detect × 5 (shake before every detection pass)
            counts_list = []
            for i in range(5):
                self._update_status_threadsafe(
                    f"Shaking... ({i + 1}/5) "
                    f"({time.time() - self._det_start_time:.1f}s)")
                if self.arduino and self.arduino.connected:
                    self.arduino.send("SHAKE")
                time.sleep(SHAKE_DURATION_S + 0.7)

                self._update_status_threadsafe(
                    f"Detecting... {i + 1}/5 "
                    f"({time.time() - self._det_start_time:.1f}s)")
                self.after(0, lambda p=(i + 1) * 20: self._det_progress.set(p))
                c = self.infer.get_counts() if self.infer else {}
                counts_list.append(c) if c else None
                time.sleep(0.5)

            self.after(0, lambda: self._det_progress.set(100))

            # 3) Take mode for each class
            final_counts = {}
            for name in CLASS_NAMES:
                vals = [c.get(name, 0) for c in counts_list if c]
                if vals:
                    final_counts[name] = Counter(vals).most_common(1)[0][0]
                else:
                    final_counts[name] = 0

            # 4) Update UI
            if mode == "initial":
                self._baseline_counts = final_counts
                for name in CLASS_NAMES:
                    self._inst_counts[name] = final_counts[name]
                    self._inst_status[name] = "present" if final_counts[name] > 0 else " "
                self.after(0, self._update_checklist_display)
                total = sum(final_counts.values())
                self.after(0, lambda: self._det_status_text.set(
                    f"Initial detection complete — {total} instruments found"))
                self.after(0, lambda: self._det_progress.set(100))
                # Enable final button
                self.after(0, lambda: self.final_btn.configure(state="normal"))
                self.after(0, lambda: self.final_btn.configure(
                    bg=ACCENT, fg=BG_DARK, activebackground="#00a88c", activeforeground=BG_DARK))

            else:  # final
                self._final_counts = final_counts
                total_missing = 0
                weight_missing = False

                for name in CLASS_NAMES:
                    baseline = self._baseline_counts.get(name, 0)
                    current = final_counts.get(name, 0)
                    self._inst_counts[name] = current
                    if current < baseline:
                        self._inst_status[name] = "MISSING"
                        total_missing += (baseline - current)
                    else:
                        self._inst_status[name] = "present"

                # Weight check: if weight dropped but camera didn't catch it
                weight_drop = self._final_weight < self._initial_weight - WEIGHT_MISSING_THRESHOLD_G
                if weight_drop and total_missing == 0:
                    weight_missing = True
                    for name in CLASS_NAMES:
                        self._inst_status[name] = "WEIGHT_MISSING"

                self.after(0, self._update_checklist_display)

                if total_missing > 0:
                    self.after(0, lambda: self.missing_card.configure(
                        text=str(total_missing), fg=DANGER))
                    self.after(0, lambda: self._det_status_text.set(
                        f"Final detection complete — {total_missing} instrument(s) MISSING"))
                elif weight_missing:
                    self.after(0, lambda: self.missing_card.configure(
                        text="WEIGHT", fg=DANGER))
                    self.after(0, lambda: self._det_status_text.set(
                        "Weight drop detected — items may be missing"))
                else:
                    self.after(0, lambda: self.missing_card.configure(text="0", fg=ACCENT))
                    self.after(0, lambda: self._det_status_text.set(
                        "Final detection complete — all items present"))

        except Exception as e:
            self.after(0, lambda: self._det_status_text.set(f"Error: {e}"))
        finally:
            self._detecting = False
            self.after(0, lambda: self._set_buttons_state("normal"))
            self.after(0, lambda: self._det_progress.set(0))

    def _update_status_threadsafe(self, text):
        self.after(0, lambda: self._det_status_text.set(text))

    # ── Toggle handlers ─────────────────────────────────

    def _on_device_toggle(self):
        online = self._dev_tog_var.get()
        self.device_card.configure(
            text="ONLINE" if online else "OFFLINE",
            fg=ACCENT if online else DANGER)

    def _on_camera_toggle(self):
        active = self._cam_tog_var.get()
        self.camera_card.configure(
            text="ACTIVE" if active else "OFFLINE",
            fg=ACCENT if active else DANGER)
        self.cam_dot.configure(fg=ACCENT if active else DANGER)

    # ── Close handler ───────────────────────────────────

    def _on_close(self):
        if self.infer:
            self.infer.stop()
        if self.arduino:
            self.arduino.disconnect()
        self.destroy()


# ── Entry point ─────────────────────────────────────────

if __name__ == "__main__":
    app = MediScanApp()
    app.mainloop()
