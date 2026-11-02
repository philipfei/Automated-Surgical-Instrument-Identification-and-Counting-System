import os
import sys
import json
import shutil
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import pycocotools.mask as maskUtils
except ImportError:
    print("FATAL: pycocotools is required for mask decoding.")
    print("Run: pip install pycocotools  (or pycocotools-windows on Windows)")
    sys.exit(1)

# ── Class filter ────────────────────────────────────────────

CLASS_GROUPS = {
    "Thumb_Forceps": [
        "Adson-Smooth-Tissue-Forceps",
        "Adson-Teeth-Tissue-Forceps",
    ],
    "Forceps": [
        "Crile-Haemostatic_Forceps",
        "FOERSTER Sponge Forceps",
        "Crile-Wood-Needle-Holder",
        "Webster-Needle-Holder",
        "forceps",
    ],
    "Scissors": [
        "Iris-Scissors",
        "Metzenbaum-Scissors",
        "Straight sharp-blunt operating",
    ],
    "Nasal_Speculum": [
        "nasal speculum",
    ],
}


def build_class_filter(coco_categories):
    """Match kept class names against COCO categories, map them into 4 groups.

    Matching is case-insensitive.  "Straight sharp-blunt operating" is
    matched as a prefix so that it resolves to the full COCO name
    "Straight sharp-blunt operating scissors".

    Returns (kept_coco_ids_set, coco_id_to_group_index, group_names_dict).
    """
    kept_coco_ids = []
    coco_id_to_group = {}
    group_names = {}
    name_to_coco_id = {cat['name']: cat['id'] for cat in coco_categories}
    lower_to_original = {n.lower(): n for n in name_to_coco_id}

    for group_idx, (group_name, class_list) in enumerate(CLASS_GROUPS.items()):
        group_names[group_idx] = group_name
        for kept_name in class_list:
            kl = kept_name.lower()

            # 1) exact case-insensitive match
            if kl in lower_to_original:
                coco_id = name_to_coco_id[lower_to_original[kl]]
                kept_coco_ids.append(coco_id)
                coco_id_to_group[coco_id] = group_idx
                continue

            # 2) prefix match ("Straight sharp-blunt operating" -> "... scissors")
            matches = [orig for low, orig in lower_to_original.items()
                       if low.startswith(kl)]
            if len(matches) == 1:
                coco_id = name_to_coco_id[matches[0]]
                kept_coco_ids.append(coco_id)
                coco_id_to_group[coco_id] = group_idx
                continue

            print(f"WARNING: could not match '{kept_name}' "
                  f"to any COCO category — skipping")

    return set(kept_coco_ids), coco_id_to_group, group_names


# ── COCO → YOLO conversion ──────────────────────────────────

def decode_segmentation_to_yolo(segmentation, width, height):
    """Decode COCO segmentation (polygon or RLE) to YOLO normalized coords."""
    polygons = []

    if isinstance(segmentation, dict) and 'counts' in segmentation:
        rle = segmentation
        if isinstance(rle['counts'], list):
            rle = maskUtils.frPyObjects([rle], height, width)[0]

        binary_mask = maskUtils.decode(rle)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) >= 3:
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                poly = approx.flatten().tolist()
                if len(poly) >= 6:
                    normalized_poly = []
                    for i in range(0, len(poly), 2):
                        x_norm = max(0.0, min(1.0, poly[i] / float(width)))
                        y_norm = max(0.0, min(1.0, poly[i + 1] / float(height)))
                        normalized_poly.append(f"{x_norm:.6f} {y_norm:.6f}")
                    polygons.append(" ".join(normalized_poly))

    elif isinstance(segmentation, list):
        for poly in segmentation:
            if len(poly) >= 6:
                normalized_poly = []
                for i in range(0, len(poly), 2):
                    x_norm = max(0.0, min(1.0, poly[i] / float(width)))
                    y_norm = max(0.0, min(1.0, poly[i + 1] / float(height)))
                    normalized_poly.append(f"{x_norm:.6f} {y_norm:.6f}")
                polygons.append(" ".join(normalized_poly))

    return polygons


def convert_coco_to_yolo_seg(raw_dir, yolo_dir, split, kept_ids, remap):
    """Convert one split from COCO JSON to YOLO segmentation format.

    Returns (class_names_dict, stats_dict).
    """
    json_path = os.path.join(raw_dir, split, "_annotations.coco.json")
    if not os.path.exists(json_path):
        print(f"  No COCO JSON found for [{split}] split, skipping.")
        return None, None

    img_out = os.path.join(yolo_dir, "images", split)
    lbl_out = os.path.join(yolo_dir, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Index annotations by image
    img_annos = {}
    for anno in coco['annotations']:
        img_annos.setdefault(anno['image_id'], []).append(anno)

    print(f"  Converting [{split}] split — {len(coco['images'])} images, "
          f"{len(coco['annotations'])} total annotations...")

    stats = {
        "total_images": 0,
        "annotated_images": 0,
        "background_images": 0,
        "skipped_annotations": 0,
        "class_counts": {idx: 0 for idx in range(len(CLASS_GROUPS))},
    }

    for img_info in coco['images']:
        img_id = img_info['id']
        fname = img_info['file_name']
        w, h = img_info['width'], img_info['height']

        stats["total_images"] += 1

        src = os.path.join(raw_dir, split, fname)
        dst = os.path.join(img_out, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            continue

        annos = img_annos.get(img_id, [])
        yolo_lines = []

        for anno in annos:
            if 'segmentation' not in anno or not anno['segmentation']:
                continue

            cat_id = anno['category_id']
            if cat_id not in kept_ids:
                stats["skipped_annotations"] += 1
                continue

            yolo_class = remap[cat_id]
            seg = anno['segmentation']
            polygons = decode_segmentation_to_yolo(seg, w, h)

            # 【修改这里】：如果该器械存在多边形，将它们合并为一行（一个实例）
            if polygons:
                merged_poly = " ".join(polygons)
                yolo_lines.append(f"{yolo_class} {merged_poly}")
                # 统计计数也应该按实例算，而不是按多边形碎片算
                stats["class_counts"][yolo_class] += 1

        txt_name = Path(fname).with_suffix('.txt')
        txt_path = os.path.join(lbl_out, txt_name)

        if yolo_lines:
            stats["annotated_images"] += 1
            with open(txt_path, 'w', encoding='utf-8') as f_out:
                f_out.write("\n".join(yolo_lines) + "\n")
        else:
            stats["background_images"] += 1
            open(txt_path, 'w').close()

    return stats


# ── YAML generation ─────────────────────────────────────────

def generate_yaml(yolo_dir, class_names):
    """Write dataset.yaml for YOLO training."""
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    # No "path:" key: Ultralytics resolves splits relative to this yaml's folder
    lines = [
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        "",
        "names:",
    ]
    for idx in sorted(class_names):
        lines.append(f"  {idx}: {class_names[idx]}")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


# ── Quality report ──────────────────────────────────────────

def print_quality_report(split_name, stats, class_names):
    """Print per-split summary with per-class instance counts."""
    print(f"\n{'=' * 60}")
    print(f"  {split_name.upper()} SPLIT — Quality Report")
    print(f"{'=' * 60}")
    print(f"  Total images:           {stats['total_images']}")
    print(f"  Annotated (kept):       {stats['annotated_images']}")
    print(f"  Background (no kept):   {stats['background_images']}")
    print(f"  Skipped (other classes):{stats['skipped_annotations']}")

    total = sum(stats['class_counts'].values())
    if total == 0:
        print("\n  (no kept-class instances in this split)")
        return

    # Per-class table
    name_width = max(len(class_names.get(i, f"class_{i}"))
                     for i in range(len(class_names)))
    print(f"\n  {'Class':<{name_width}}  Instances")
    print(f"  {'-' * name_width}  ---------")
    for i in sorted(stats['class_counts']):
        count = stats['class_counts'][i]
        name = class_names.get(i, f"class_{i}")
        print(f"  {name:<{name_width}}  {count}")
    print(f"  {'-' * name_width}  ---------")
    print(f"  {'TOTAL':<{name_width}}  {total}")


# ── Background negatives ────────────────────────────────────

def add_background_negatives(backgrounds_dir, train_img_dir, train_lbl_dir):
    """Copy GoPro background frames into the training set as negative samples.

    Each background image gets an empty YOLO label file, teaching the model
    that complex real-world backgrounds contain no instruments.
    """
    if not os.path.isdir(backgrounds_dir):
        print("\nNo backgrounds/ directory found, skipping background negatives.")
        return

    valid_exts = {'.jpg', '.jpeg', '.png'}
    bg_files = [f for f in os.listdir(backgrounds_dir)
                if Path(f).suffix.lower() in valid_exts]
    if not bg_files:
        print("\nbackgrounds/ is empty, skipping background negatives.")
        print("  To suppress false positives on GoPro, add background frames")
        print("  to backgrounds/ and re-run this script.")
        return

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)

    added = 0
    for fname in bg_files:
        src = os.path.join(backgrounds_dir, fname)
        dst_name = f"bg_{fname}"
        dst_img = os.path.join(train_img_dir, dst_name)
        if not os.path.exists(dst_img):
            shutil.copy(src, dst_img)
        dst_label = os.path.join(train_lbl_dir, Path(dst_name).with_suffix('.txt'))
        if not os.path.exists(dst_label):
            open(dst_label, 'w').close()
        added += 1

    print(f"\nAdded {added} background images as negative training samples.")


# ── Raw directory discovery ─────────────────────────────────

def find_raw_dir():
    """Auto-discover the downloaded MDP dataset directory."""
    candidates = []
    data_dir = ROOT / "data"
    if data_dir.is_dir():
        for entry in os.listdir(data_dir):
            full = os.path.join(data_dir, entry)
            if os.path.isdir(full) and entry.lower().startswith("mdp-dataset"):
                candidates.append(full)

    if candidates:
        # Return the most recently modified
        candidates.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        return candidates[0]
    return None


# ── Main ────────────────────────────────────────────────────

def main(raw_dir=None):
    # Resolve raw dataset directory
    if raw_dir is None:
        raw_dir = find_raw_dir()
    if raw_dir is None or not os.path.isdir(raw_dir):
        print("ERROR: Could not find the downloaded MDP dataset.")
        print("Run data_download.py first, then pass the path:")
        print("  python scripts/data_prepare.py <path-to-MDP-dataset>")
        sys.exit(1)

    raw_dir = os.path.abspath(raw_dir)
    print(f"Raw dataset: {raw_dir}")

    # Read categories from the train split COCO JSON
    train_json = os.path.join(raw_dir, "train", "_annotations.coco.json")
    if not os.path.exists(train_json):
        print(f"ERROR: COCO JSON not found at {train_json}")
        sys.exit(1)

    with open(train_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    kept_ids, remap, kept_names = build_class_filter(coco['categories'])
    print(f"\nClass filter: keeping {len(kept_ids)} / {len(coco['categories'])} classes")
    for idx in sorted(kept_names):
        print(f"  [{idx}] {kept_names[idx]}")

    target_yolo_dir = os.path.join(os.path.dirname(raw_dir), "yolo_surgical_dataset")
    if os.path.exists(target_yolo_dir):
        print(f"\nWARNING: Output directory exists, files will be overwritten:")
        print(f"  {target_yolo_dir}")

    print("\n" + "=" * 60)
    print("  COCO → YOLO Segmentation Conversion")
    print("=" * 60)

    train_stats = convert_coco_to_yolo_seg(raw_dir, target_yolo_dir, "train", kept_ids, remap)
    valid_stats = convert_coco_to_yolo_seg(raw_dir, target_yolo_dir, "valid", kept_ids, remap)
    test_stats = convert_coco_to_yolo_seg(raw_dir, target_yolo_dir, "test", kept_ids, remap)

    # Print reports
    if train_stats:
        print_quality_report("TRAIN", train_stats, kept_names)
    if valid_stats:
        print_quality_report("VALID", valid_stats, kept_names)
    if test_stats:
        print_quality_report("TEST", test_stats, kept_names)

    # Generate YAML
    if train_stats:
        generate_yaml(target_yolo_dir, kept_names)
        print(f"\nDataset YAML written to: {os.path.join(target_yolo_dir, 'dataset.yaml')}")

    # Background negatives
    add_background_negatives(
        str(ROOT / "data" / "backgrounds"),
        os.path.join(target_yolo_dir, "images", "train"),
        os.path.join(target_yolo_dir, "labels", "train"),
    )

    # Summary
    total_kept = (train_stats['annotated_images'] if train_stats else 0)
    print(f"\n{'=' * 60}")
    print(f"  Conversion complete.")
    print(f"  Output: {target_yolo_dir}")
    print(f"  Classes kept: {len(kept_names)}")
    print(f"  Training images with kept annotations: {total_kept}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else None
    main(raw)
