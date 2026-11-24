import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description="Inspect Kaggle facial keypoint labels.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    counts = Counter()
    example = None
    total = valid_images = complete = 0
    with (args.data_dir / "training.csv").open(newline="") as file:
        rows = csv.DictReader(file)
        landmarks = [name[:-2] for name in rows.fieldnames if name.endswith("_x")]
        for row in rows:
            total += 1
            if not row["Image"]:
                continue
            valid_images += 1
            points = {
                name: (float(row[f"{name}_x"]), float(row[f"{name}_y"]))
                for name in landmarks
                if row[f"{name}_x"] and row[f"{name}_y"]
            }
            counts.update(points.keys())
            if len(points) == len(landmarks):
                complete += 1
                if example is None:
                    example = (row["Image"], points)

    if example is None:
        raise ValueError("No image with all landmarks was found")

    pixels, points = example
    image = Image.frombytes("L", (96, 96), bytes(map(int, pixels.split())))
    scale = 5
    image = image.resize((96 * scale, 96 * scale), Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y in points.values():
        x, y = (x + 0.5) * scale, (y + 0.5) * scale
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#5fffe0", outline="#112822", width=2)
    image.save(args.output)
    print(f"{total} training rows, {valid_images} with images, {complete} with all landmarks.")
    for name, count in counts.most_common():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
