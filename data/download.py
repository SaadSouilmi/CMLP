# /// script
# dependencies = ["polars", "pillow"]
# ///
"""MNIST binary format:

Header (16 bytes): char[4] "MNST", uint32 version=1, uint32 count, uint32 features=784.
Each record (3140 bytes): float32 pixels[784], uint32 label (0..9).
Pixels are row-major: pixels[y * 28 + x] = grayscale(x, y) / 255.0.
"""

from io import BytesIO
from pathlib import Path
import struct

import polars as pl
from PIL import Image


def save_split(split):
    url = f"hf://datasets/ylecun/mnist/mnist/{split}-00000-of-00001.parquet"
    df = pl.read_parquet(url)
    path = Path(__file__).resolve().parent / f"{split}.bin"
    record = struct.Struct("<784fI")

    with path.open("wb") as output:
        output.write(struct.pack("<4sIII", b"MNST", 1, len(df), 784))
        for image_data, label in df.select("image", "label").iter_rows():
            with Image.open(BytesIO(image_data["bytes"])) as image:
                if image.size != (28, 28):
                    raise ValueError(f"Unexpected image size: {image.size}")
                pixels = image.convert("L").tobytes()
            if not 0 <= label <= 9:
                raise ValueError(f"Invalid label: {label}")
            vector = [pixel / 255.0 for pixel in pixels]
            output.write(record.pack(*vector, label))

    print(f"Saved {len(df):,} samples to {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    for split in ("train", "test"):
        save_split(split)
