from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.config import DATA_DIR


def main() -> None:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    bad_images: list[tuple[Path, str]] = []

    for image_path in sorted(DATA_DIR.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
            continue
        try:
            image_bytes = tf.io.read_file(str(image_path))
            tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        except Exception as exc:
            bad_images.append((image_path, str(exc).splitlines()[0]))

    if not bad_images:
        print("No TensorFlow decoder issues found.")
        return

    print(f"Found {len(bad_images)} TensorFlow-unreadable image(s):")
    for image_path, error in bad_images:
        print(f"{image_path}\t{error}")


if __name__ == "__main__":
    main()
