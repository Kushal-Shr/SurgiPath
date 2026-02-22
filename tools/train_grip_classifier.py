"""
Grip Classifier Training Script for MedLab Coach AI
=====================================================

HOW TO USE:
  1. First collect data: streamlit run tools/collect_grip_data.py
  2. Then train:        python tools/train_grip_classifier.py
  3. Model saved to:    models/grip_classifier.joblib

HOW IT WORKS:
  - Reads data/grip_samples.csv (created by the collection tool)
  - Trains a RandomForest classifier on 14 geometric hand features
  - Features are scale-invariant (normalized by palm size)
  - Uses 80/20 train/test split with stratified sampling
  - Prints accuracy, confusion matrix, per-class metrics
  - Saves the trained model as a joblib file

FEATURES USED (14 total):
  - 4 finger-to-thumb distances (normalized)
  - 4 finger curl ratios
  - 3 inter-finger spreads
  - grip angle (thumb-index angle at wrist)
  - instrument angle (wrist-to-fingertip vs horizontal)
  - palm size (absolute, for reference)

The trained model is loaded by src/hands.py at runtime. If the model
file doesn't exist, the app falls back to geometric heuristics.
"""

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "grip_samples.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "grip_classifier.joblib"

FEATURE_NAMES = [
    "thumb_index_dist", "thumb_middle_dist", "thumb_ring_dist", "thumb_pinky_dist",
    "index_curl", "middle_curl", "ring_curl", "pinky_curl",
    "index_middle_spread", "middle_ring_spread", "ring_pinky_spread",
    "grip_angle", "instrument_angle", "palm_size",
]


def load_data():
    if not DATA_PATH.exists():
        print(f"ERROR: No data found at {DATA_PATH}")
        print("Run the collection tool first: streamlit run tools/collect_grip_data.py")
        sys.exit(1)

    X, y = [], []
    with open(DATA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = []
            valid = True
            for fn in FEATURE_NAMES:
                try:
                    features.append(float(row[fn]))
                except (KeyError, ValueError):
                    valid = False
                    break
            if valid:
                X.append(features)
                y.append(row["grip_type"])

    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("Grip Classifier Training")
    print("=" * 60)

    X, y = load_data()
    print(f"\nLoaded {len(X)} samples from {DATA_PATH}")

    classes, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(classes, counts):
        print(f"  {cls}: {cnt} samples")

    if len(X) < 20:
        print(f"\nWARNING: Only {len(X)} samples. Recommend at least 50 per class.")
        print("The model will train but may not be accurate.")

    if len(classes) < 2:
        print("\nERROR: Need at least 2 different grip types to train a classifier.")
        sys.exit(1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    min_class_count = min(counts)
    test_size = 0.2 if min_class_count >= 5 else 0.1
    if min_class_count < 3:
        print("\nWARNING: Some classes have <3 samples. Using leave-one-out instead of split.")
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y,
        )

    # Train
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"\n{'=' * 40}")
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"{'=' * 40}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=classes))

    # Cross-validation if enough data
    if min_class_count >= 5:
        cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, min_class_count), scoring="accuracy")
        print(f"\nCross-validation accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

    # Feature importance
    print(f"\nTop 5 most important features:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]}: {importances[idx]:.3f}")

    # Save model + scaler together
    model_bundle = {
        "classifier": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "classes": list(classes),
        "accuracy": float(accuracy),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Classes: {list(classes)}")
    print(f"\nTo use in the app, just restart streamlit — src/hands.py will auto-load the model.")


if __name__ == "__main__":
    main()
