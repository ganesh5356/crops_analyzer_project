from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR, REPORT_DIR
from src.data import count_images_by_class, validate_dataset_dir


def main() -> None:
    validate_dataset_dir(DATA_DIR)
    REPORT_DIR.mkdir(exist_ok=True)

    counts = count_images_by_class(DATA_DIR)
    overview = pd.DataFrame(
        [{"class_name": class_name, "image_count": count} for class_name, count in counts.items()]
    ).sort_values("class_name")
    output_path = REPORT_DIR / "dataset_overview.csv"
    overview.to_csv(output_path, index=False)

    print(f"Classes: {len(overview)}")
    print(f"Images: {overview['image_count'].sum()}")
    print(overview.to_string(index=False))
    print(f"Saved overview: {output_path}")


if __name__ == "__main__":
    main()
