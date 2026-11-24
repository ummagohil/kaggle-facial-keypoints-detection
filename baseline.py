import argparse
import csv
import math
import random
from pathlib import Path
from statistics import mean


def mean_locations(labels, features):
    return {
        name: mean(row[name] for row in labels if name in row)
        for name in features
    }


def write_submission(data_dir, output, predict_location):
    with (data_dir / "IdLookupTable.csv").open(newline="") as file:
        requests = list(csv.DictReader(file))
    with (data_dir / "SampleSubmission.csv").open(newline="") as file:
        sample = csv.DictReader(file)
        columns = ["RowId", "Location"]
        if sample.fieldnames != columns:
            raise ValueError("Unexpected sample submission columns")
        expected_ids = [row["RowId"] for row in sample]

    row_ids = [row["RowId"] for row in requests]
    if not row_ids or row_ids != expected_ids or len(set(row_ids)) != len(row_ids):
        raise ValueError("Lookup RowIds must match the sample submission in order")

    predictions = []
    for row in requests:
        location = predict_location(int(row["ImageId"]), row["FeatureName"])
        if not math.isfinite(location) or not 0 <= location <= 96:
            raise ValueError(f"Invalid pixel coordinate for RowId {row['RowId']}: {location}")
        predictions.append((row["RowId"], location))

    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(predictions)
    return len(predictions)


def main():
    parser = argparse.ArgumentParser(description="Score a mean facial-landmark baseline.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--submission", type=Path, help="Write Kaggle submission CSV")
    args = parser.parse_args()

    with (args.data_dir / "training.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        features = [name for name in reader.fieldnames if name != "Image"]
        labels = [
            {name: float(row[name]) for name in features if row[name]}
            for row in reader
            if row["Image"]
        ]

    # Split by image so its x and y coordinates stay in the same partition.
    random.Random(42).shuffle(labels)
    split = int(0.8 * len(labels))
    train, validation = labels[:split], labels[split:]
    predictions = mean_locations(train, features)
    squared_errors = [
        (value - predictions[name]) ** 2
        for row in validation
        for name, value in row.items()
    ]
    print(f"Training images: {len(train):,}")
    print(f"Validation images: {len(validation):,}")
    print(f"Labeled validation coordinates: {len(squared_errors):,}")
    print(f"Validation RMSE: {math.sqrt(mean(squared_errors)):.3f} pixels")

    if args.submission is not None:
        # Refit on all labeled images after measuring the held-out score.
        submission_means = mean_locations(labels, features)
        count = write_submission(
            args.data_dir,
            args.submission,
            lambda _image_id, feature: submission_means[feature],
        )
        print(f"Wrote {count:,} predictions to {args.submission}")


if __name__ == "__main__":
    main()
