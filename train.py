from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    METADATA_PATH,
    DATA_DIR,
    HISTORY_PATH,
    IMAGE_SIZE,
    MODEL_DIR,
    MODEL_PATH,
    PLOT_PATH,
    REPORT_DIR,
    SEED,
)
from src.data import count_images_by_class, load_datasets, make_class_weights
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sentinel-2 satellite image classifier.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Folder containing class subfolders.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0], help="Square image size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--arch",
        choices=["mobilenetv2", "resnet50", "efficientnetb0"],
        default="mobilenetv2",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--base-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet pretrained weights or initialize base model randomly.",
    )
    return parser.parse_args()


def plot_history(history: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    data_dir = Path(args.data_dir)
    image_size = (args.image_size, args.image_size)
    train_ds, val_ds, class_names = load_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    class_counts = count_images_by_class(data_dir)
    class_weights = make_class_weights(class_counts, class_names)

    base_weights = None if args.base_weights == "none" else args.base_weights
    model = build_model(
        num_classes=len(class_names),
        image_size=image_size,
        arch=args.arch,
        base_weights=base_weights,
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    model.save(MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    metadata = {
        "arch": args.arch,
        "image_size": image_size,
        "class_names": class_names,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_PATH, index=False)
    plot_history(history_df)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved class labels: {CLASS_NAMES_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved training history: {HISTORY_PATH}")
    print(f"Saved curves: {PLOT_PATH}")


if __name__ == "__main__":
    main()
