

import os
import cv2
import argparse
import warnings
import numpy as np
import joblib
from tqdm import tqdm
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  (tweak these if needed)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = 96      # Resize for HOG + color feature extraction
WIN_SIZE    = 96      # Sliding window size (must equal IMG_SIZE)
STRIDE      = 24      # Step size — smaller = more thorough but slower
CONF_THRESH = 0.65    # Min confidence to show a detection
NMS_IOU     = 0.30    # NMS overlap threshold
SCALES      = [1.0, 0.75, 0.5]   # Multi-scale pyramid
MODEL_FILE  = "flower_svm_model.pkl"


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTOR  — HOG + HSV Color Histogram
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(img_bgr):
    """
    Combine HOG (shape) + HSV color histogram (flower color).
    Returns a fixed-length float32 feature vector.
    """
    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # HOG on grayscale
    gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        
        block_norm='L2-Hys',
        transform_sqrt=True,
        visualize=False
    )

    # HSV color histogram (captures flower hue and saturation)
    hsv    = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-6)
    s_hist = s_hist / (s_hist.sum() + 1e-6)

    return np.concatenate([hog_feat, h_hist, s_hist]).astype(np.float32)


def augment_image(img_bgr):
    """
    Create lightweight training augmentations while keeping the flower label.
    These improve robustness to camera angle and lighting changes.
    """
    augmented = [cv2.flip(img_bgr, 1)]

    h, w = img_bgr.shape[:2]
    center = (w / 2, h / 2)
    for angle in (-12, 12):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img_bgr,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        augmented.append(rotated)

    augmented.append(cv2.convertScaleAbs(img_bgr, alpha=1.12, beta=12))
    augmented.append(cv2.convertScaleAbs(img_bgr, alpha=0.88, beta=-12))

    return augmented


def class_confidences(model, feat):
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            proba = model.predict_proba(feat)[0]
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
        total = proba.sum()
        if total > 0:
            return proba / total

    scores = model.decision_function(feat)
    scores = np.atleast_2d(scores).astype(np.float64)[0]
    if scores.size == 1:
        scores = np.array([-scores[0], scores[0]], dtype=np.float64)
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / (exp_scores.sum() + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(dataset_path):
    """
    Load images from class sub-folders and extract features.
    Adds background patches as a negative class.
    """
    X, y = [], []
    X_img, y_img = [], []
    rng  = np.random.default_rng(42)

    class_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    print(f"\nClasses found: {class_dirs}")

    for cls in class_dirs:
        cls_dir = os.path.join(dataset_path, cls)
        files   = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"  [{cls}] - {len(files)} images")

        for fname in tqdm(files, desc=f"  {cls}", leave=False):
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(cls)
            X_img.append(feat)
            y_img.append(cls)
            for aug in augment_image(img):
                X.append(extract_features(aug))
                y.append(cls)

    # ── Background (negative) patches ────────────────────────────────────────
    print("  Mining background patches...")
    bg = 0
    for cls in class_dirs:
        cls_dir = os.path.join(dataset_path, cls)
        files   = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        sample = rng.choice(files, size=min(40, len(files)), replace=False)
        for fname in sample:
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            for _ in range(2):
                px = int(rng.integers(0, max(1, w - WIN_SIZE)))
                py = int(rng.integers(0, max(1, h - WIN_SIZE)))
                patch = img[py:py + WIN_SIZE, px:px + WIN_SIZE]
                if patch.shape[:2] == (WIN_SIZE, WIN_SIZE):
                    if py < h * 0.25 or py > h * 0.75:
                        X.append(extract_features(patch))
                        y.append("background")
                        bg += 1

    print(f"  Background patches: {bg}")

    X  = np.array(X, dtype=np.float32)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\nTotal samples : {len(X)}")
    print(f"Feature size  : {X.shape[1]}")
    for cls, cnt in zip(*np.unique(y, return_counts=True)):
        print(f"  {cls:<14}: {cnt}")

    X_img = np.array(X_img, dtype=np.float32)
    y_img_enc = le.transform(y_img)

    return X, y_enc, le, X_img, y_img_enc


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def train(dataset_path):
    print("\n" + "=" * 55)
    print("  TRAINING - HOG + Color + Linear Classifier Flower Detector")
    print("=" * 55)

    X, y_enc, le, X_img, y_img_enc = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")
    print("Training SVM classifier... (usually a few minutes on full dataset)")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            class_weight='balanced',
            random_state=42,
            max_iter=-1
        ))
    ])
    model.fit(X_train, y_train)

    image_model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='cosine'
        ))
    ])
    image_model.fit(X_img, y_img_enc)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\nTraining complete!")
    print(f"   Test Accuracy : {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump({"model": model, "image_model": image_model, "le": le}, MODEL_FILE)
    print(f"Model saved -> {MODEL_FILE}")

    return model, image_model, le


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sliding_window(image_bgr, model, le):
    orig_h, orig_w = image_bgr.shape[:2]
    detections = []

    for scale in SCALES:
        sw = int(orig_w * scale)
        sh = int(orig_h * scale)
        if sw < WIN_SIZE or sh < WIN_SIZE:
            continue
        scaled = cv2.resize(image_bgr, (sw, sh))

        for yp in range(0, sh - WIN_SIZE + 1, STRIDE):
            for xp in range(0, sw - WIN_SIZE + 1, STRIDE):
                crop = scaled[yp:yp + WIN_SIZE, xp:xp + WIN_SIZE]
                feat = extract_features(crop).reshape(1, -1)

                proba    = class_confidences(model, feat)
                idx      = int(np.argmax(proba))
                conf     = float(proba[idx])
                cls_name = le.classes_[idx]

                if conf >= CONF_THRESH and cls_name != "background":
                    bx = int(xp / scale)
                    by = int(yp / scale)
                    bw = int(WIN_SIZE / scale)
                    bh = int(WIN_SIZE / scale)
                    detections.append((bx, by, bw, bh, cls_name, conf))

    return detections


def nms(detections):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[5], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        bx, by, bw, bh = best[:4]
        rest = []
        for d in detections:
            dx, dy, dw, dh = d[:4]
            ix  = max(0, min(bx + bw, dx + dw) - max(bx, dx))
            iy  = max(0, min(by + bh, dy + dh) - max(by, dy))
            iou = (ix * iy) / (bw * bh + dw * dh - ix * iy + 1e-6)
            if iou < NMS_IOU:
                rest.append(d)
        detections = rest
    return kept


def merge_boxes(detections, shape):
    ih, iw = shape[:2]
    groups  = defaultdict(list)
    for d in detections:
        groups[d[4]].append(d)
    merged = []
    for cls, ds in groups.items():
        x1 = max(0,  min(d[0]        for d in ds))
        y1 = max(0,  min(d[1]        for d in ds))
        x2 = min(iw, max(d[0] + d[2] for d in ds))
        y2 = min(ih, max(d[1] + d[3] for d in ds))
        merged.append((x1, y1, x2 - x1, y2 - y1, cls, max(d[5] for d in ds)))
    return merged


def select_single_detection(image_bgr, model, le, detections, image_model=None):
    feat = extract_features(image_bgr).reshape(1, -1)

    if image_model is not None:
        pred_idx = int(image_model.predict(feat)[0])
        image_cls = le.classes_[pred_idx]
        image_conf = float(np.max(image_model.predict_proba(feat)[0]))
    else:
        proba = class_confidences(model, feat)
        order = np.argsort(proba)[::-1]

        image_cls = None
        image_conf = 0.0
        for idx in order:
            cls_name = le.classes_[idx]
            if cls_name != "background":
                image_cls = cls_name
                image_conf = float(proba[idx])
                break

    if image_cls is None:
        return []

    matching = [d for d in detections if d[4] == image_cls]
    if matching:
        best = max(matching, key=lambda d: (d[5], d[2] * d[3]))
        x, y, w, h, _, box_conf = best
        return [(x, y, w, h, image_cls, max(image_conf, box_conf))]

    ih, iw = image_bgr.shape[:2]
    return [(0, 0, iw, ih, image_cls, image_conf)]


# ─────────────────────────────────────────────────────────────────────────────
#  DETECT
# ─────────────────────────────────────────────────────────────────────────────
def detect(image_path, model=None, le=None, show=True, image_model=None):
    # Load saved model if not provided
    if model is None or le is None:
        if not os.path.exists(MODEL_FILE):
            print(f"Model not found: {MODEL_FILE}")
            print("   Train first:  python flower_detection.py --dataset flowers_annotated --train")
            return
        saved = joblib.load(MODEL_FILE)
        model = saved["model"]
        image_model = saved.get("image_model")
        le    = saved["le"]
        print(f"Model loaded from {MODEL_FILE}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    # Resize large images for speed
    oh, ow = img.shape[:2]
    if max(oh, ow) > 600:
        sc  = 600 / max(oh, ow)
        img = cv2.resize(img, (int(ow * sc), int(oh * sc)))

    print(f"Detecting in {image_path}  ({img.shape[1]}x{img.shape[0]} px)")
    print("   Running sliding window (30–60 sec)...")

    raw   = sliding_window(img, model, le)
    after = nms(raw)
    final = select_single_detection(img, model, le, after, image_model=image_model)

    print(f"   Raw: {len(raw)}  -> NMS: {len(after)}  -> Final: {len(final)}")

    # Draw RED bounding boxes on result image
    out = img.copy()
    for (x, y, w, h, cls, conf) in final:
        label        = f"{cls}  {conf:.0%}"
        RED          = (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), RED, 3)
        (tw, th), _  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(out, (x, y - th - 12), (x + tw + 8, y), RED, -1)
        cv2.putText(out, label, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Side-by-side: original | detected
    h_out = 400
    w_out = int(img.shape[1] * h_out / img.shape[0])
    left  = cv2.resize(img, (w_out, h_out))
    right = cv2.resize(out, (w_out, h_out))
    combined = np.hstack([left, right])

    cv2.putText(combined, "Original",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    title = f"Detected: {final[0][4]}" if final else "Detected: 0 flower(s)"
    cv2.putText(combined, title,
                (w_out + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Save result
    result_path = "detection_result.jpg"
    cv2.imwrite(result_path, combined)
    print(f"\nResult saved -> {result_path}")

    # Print summary
    if final:
        x, y, w, h, cls, conf = final[0]
        print("\nDetection:")
        print(f"  {cls:<14} | {conf:.1%} confidence | Box: ({x},{y}) {w}x{h}px")
    else:
        print("\nNo flower detected.")
        print("   Try lowering CONF_THRESH at the top of the file (e.g. 0.55)")

    if show:
        cv2.imshow("Flower Detection - press any key to close", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Flower Object Detection - HOG + Linear Classifier")
    parser.add_argument("--dataset", type=str, default="flowers_annotated",
                        help="Path to dataset folder (with class sub-folders)")
    parser.add_argument("--train",   action="store_true",
                        help="Train the SVM model on the dataset")
    parser.add_argument("--image",   type=str, default=None,
                        help="Path to an image to detect flowers in")
    parser.add_argument("--no-display", action="store_true",
                        help="Save detection result without opening a window")
    args = parser.parse_args()

    model, image_model, le = None, None, None

    if args.train:
        if not os.path.isdir(args.dataset):
            print(f"Dataset folder not found: {args.dataset}")
            return
        model, image_model, le = train(args.dataset)

    if args.image:
        detect(args.image, model, le, show=not args.no_display, image_model=image_model)
    elif not args.train:
        print("\n  HOW TO USE:")
        print("  -----------------------------------------------------")
        print("  Train:   python flower_detection.py --dataset flowers_annotated --train")
        print("  Detect:  python flower_detection.py --image rose.jpg")
        print("  Both:    python flower_detection.py --dataset flowers_annotated --train --image rose.jpg")
        print("  -----------------------------------------------------")


if __name__ == "__main__":
    main()
