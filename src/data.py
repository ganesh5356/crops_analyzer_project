from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.config import BATCH_SIZE, IMAGE_SIZE, SEED


def validate_dataset_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    class_dirs = [path for path in data_dir.iterdir() if path.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least 2 class folders inside {data_dir}, found {len(class_dirs)}."
        )


def load_datasets(
    data_dir: Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.2,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    validate_dataset_dir(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1024, seed=seed).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def count_images_by_class(data_dir: Path) -> dict[str, int]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts: dict[str, int] = {}
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1 for path in class_dir.rglob("*") if path.suffix.lower() in image_exts
        )
    return counts


def make_class_weights(class_counts: dict[str, int], class_names: list[str]) -> dict[int, float]:
    total = sum(class_counts.values())
    num_classes = len(class_names)
    return {
        index: total / (num_classes * max(class_counts[class_name], 1))
        for index, class_name in enumerate(class_names)
    }
