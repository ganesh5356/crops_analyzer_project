from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CLASS_NAMES_PATH, MODEL_PATH, METADATA_PATH


def model_image_size(model: tf.keras.Model) -> tuple[int, int]:
    _, height, width, _ = model.input_shape
    return int(height), int(width)


def load_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(image_size)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict classes for Sentinel-2 satellite images.")
    parser.add_argument("image", help="Path to a satellite image or directory of images.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to show (for single image).")
    parser.add_argument("--output", type=str, default=None, help="Path to export predictions (CSV or JSON).")
    args = parser.parse_args()

    input_path = Path(args.image)
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    if input_path.is_dir():
        image_paths = sorted([
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ])
        if not image_paths:
            print(f"No images found in directory: {input_path}")
            return
        is_batch = True
    else:
        image_paths = [input_path]
        is_batch = False

    # Load model and class names
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    image_size = model_image_size(model)

    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        class_names = metadata.get("class_names", [])
    elif CLASS_NAMES_PATH.exists():
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("Class names / metadata not found.")

    results = []

    if is_batch:
        print(f"Classifying {len(image_paths)} images...")
        print(f"{'Image Path':<50} | {'Top Predicted Class':<20} | {'Confidence':<10}")
        print("-" * 88)
        
        for path in image_paths:
            try:
                probabilities = model.predict(load_image(path, image_size), verbose=0)[0]
                top_idx = probabilities.argmax()
                top_label = class_names[top_idx]
                top_score = float(probabilities[top_idx])
                
                # Print row
                path_str = str(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)
                # Truncate path if too long
                if len(path_str) > 47:
                    path_str = "..." + path_str[-44:]
                print(f"{path_str:<50} | {top_label:<20} | {top_score:.2%}")
                
                # Store for exporting
                all_preds = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                results.append({
                    "image_path": str(path),
                    "predicted_class": top_label,
                    "confidence": top_score,
                    "predictions": all_preds
                })
            except Exception as e:
                print(f"Error predicting {path.name}: {e}")
    else:
        # Single image
        path = image_paths[0]
        probabilities = model.predict(load_image(path, image_size), verbose=0)[0]
        top_indices = probabilities.argsort()[-args.top_k:][::-1]
        
        print(f"\nPredictions for {path.name}:")
        for idx in top_indices:
            label = class_names[idx]
            score = float(probabilities[idx])
            print(f"  {label:<20}: {score:.2%}")
            
        all_preds = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        results.append({
            "image_path": str(path),
            "predicted_class": class_names[top_indices[0]],
            "confidence": float(probabilities[top_indices[0]]),
            "predictions": all_preds
        })

    # Export results if output path is specified
    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".json":
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nSaved predictions to JSON: {out_path}")
        elif out_path.suffix.lower() == ".csv":
            with open(out_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "predicted_class", "confidence"])
                for r in results:
                    writer.writerow([r["image_path"], r["predicted_class"], f"{r['confidence']:.4f}"])
            print(f"\nSaved predictions to CSV: {out_path}")
        else:
            print(f"\nUnknown format: {out_path.suffix}. Please specify .csv or .json for export.")


if __name__ == "__main__":
    main()
