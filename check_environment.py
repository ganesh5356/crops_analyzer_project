from __future__ import annotations

import platform
import struct
import sys


def main() -> None:
    version = sys.version_info
    architecture = struct.calcsize("P") * 8
    print(f"Python: {platform.python_version()}")
    print(f"Architecture: {architecture}-bit")
    print(f"Platform: {platform.system()}")

    if platform.system() == "Windows" and (version.major, version.minor) not in {
        (3, 11),
        (3, 12),
    }:
        raise SystemExit(
            "This project needs 64-bit Python 3.11 or 3.12 on Windows for "
            "TensorFlow/TensorFlow-Intel. Recreate .venv with Python 3.12."
        )

    if architecture != 64:
        raise SystemExit("This project needs 64-bit Python.")

    print("Environment looks compatible.")


if __name__ == "__main__":
    main()
