# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated Surgical Instrument Identification and Counting System. A YOLO11m-seg instance segmentation model (base `yolo11m-seg.pt`) detects and counts 4 instrument classes (Thumb_Forceps, Forceps, Scissors, Nasal_Speculum) from a GoPro stream; a Tkinter GUI compares before/after counts, drives an Arduino shaker tray and cross-checks with load-cell weight. Trained on the Roboflow MDP dataset. GitHub: https://github.com/philipfei/Automated-Surgical-Instrument-Identification-and-Counting-System

## Commands

Run everything from the repo root. Every script resolves paths from `ROOT = Path(__file__).resolve().parents[1]`, so cwd does not matter for data/weights paths.

```bash
pip install -r requirements.txt   # Windows: pycocotools-windows instead of pycocotools
```

- **Download dataset**: `python scripts/data_download.py` — needs `ROBOFLOW_API_KEY` env var; downloads to `data/MDP-Dataset_accessary-2/`
- **Prepare dataset**: `python scripts/data_prepare.py [raw_dataset_dir]` — merges raw classes into 4 groups (`CLASS_GROUPS`), COCO→YOLO-seg, writes `dataset.yaml` (no `path:` key, so it is portable), copies `data/backgrounds/` into train as `bg_*` negatives
- **Train**: `python scripts/train.py` — 60 epochs, 640px, batch 16, device 0, aggressive augmentation; then evaluates on the test split. Checks `C:/YOLO_Data/yolo_surgical_dataset/dataset.yaml` (lab training machine) before the in-repo dataset
- **Resume training**: `python scripts/resume.py`
- **Batch predict**: `python scripts/predict.py` — images in `data/predict/`, output to `data/predict/output/`
- **GUI (full system)**: `python app/ui.py`
- **GoPro standalone viewer**: `python app/gopro_inference.py` — GoPro at `172.23.111.51`, UDP stream on port 8554

## Architecture

```
scripts/   offline pipeline: data_download → data_prepare → train/resume → predict
app/       gopro_inference.py (GoProInference: capture + inference threads, get_frame/get_counts)
           ui.py (MediScanApp GUI + ArduinoLink serial thread; imports gopro_inference from same dir)
firmware/tray_controller/tray_controller.ino   4× HX711 load cells + shaker motor
data/      MDP-Dataset_accessary-2 (raw COCO), yolo_surgical_dataset (YOLO), backgrounds, predict
runs/segment/train/weights/best.pt   model used by app/ and predict.py
```

**Serial protocol** (115200 baud): GUI sends `RESET` / `SHAKE` / `STOP` / `WEIGHT`; Arduino replies `OK:<CMD>` and streams `Filtered:<grams>`. `SHAKE_DURATION` in the firmware must match `SHAKE_DURATION_S` in `app/ui.py`.

**Detection workflow** (`MediScanApp._run_detection`): record weight → 5× (shake → read counts) → per-class mode. Final vs initial: lower count = MISSING; weight drop > `WEIGHT_MISSING_THRESHOLD_G` with no visual miss = weight alert.

## Known Issues

- No tests or linting configuration.
- `last.pt` and root-level `*.pt` base weights are gitignored; only `best.pt` is versioned.
- Domain shift: training data has plain backgrounds; GoPro inference has complex backgrounds. Augmentation (rotation, mixup, copy_paste) and background negatives are meant to bridge this gap.
""
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
