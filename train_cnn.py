import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline import write_submission


def load_faces(path, features=()):
    images, coordinates, image_ids = [], [], []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if not row["Image"]:
                continue
            images.append(np.fromstring(row["Image"], dtype=np.uint8, sep=" ").reshape(1, 96, 96))
            coordinates.append([float(row[name]) if row[name] else np.nan for name in features])
            image_ids.append(int(row.get("ImageId", len(images))))
    return torch.from_numpy(np.stack(images)), torch.tensor(coordinates, dtype=torch.float32), image_ids


def flip_indices(features):
    indices = {name: index for index, name in enumerate(features)}
    # Eye/eyebrow names start with the side. Mouth corner names put it in the middle.
    opposite = {"left": "right", "right": "left"}
    return [indices["_".join(opposite.get(part, part) for part in name.split("_"))] for name in features]


def make_model(outputs):
    return nn.Sequential(
        nn.AvgPool2d(2),
        nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, outputs),
    )


def squared_error(predictions, targets, known):
    # Replace missing targets before subtraction so NaNs never enter backpropagation.
    return ((predictions - targets).square() * known).sum()


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on available facial coordinates.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with (args.data_dir / "training.csv").open(newline="") as file:
        features = [name for name in next(csv.reader(file)) if name != "Image"]
    images, raw_labels, _ = load_faces(args.data_dir / "training.csv", features)
    indices = list(range(len(images)))
    random.Random(42).shuffle(indices)
    split = int(0.8 * len(indices))
    known = raw_labels.isfinite()
    labels = ((raw_labels - 48) / 48).nan_to_num()
    train = DataLoader(TensorDataset(images[indices[:split]], labels[indices[:split]],
                                     known[indices[:split]]), batch_size=128, shuffle=True)
    val = DataLoader(TensorDataset(images[indices[split:]], labels[indices[split:]],
                                   known[indices[split:]]), batch_size=128)
    swap = torch.tensor(flip_indices(features), device=device)
    x_axis = torch.tensor([name.endswith("_x") for name in features], device=device)
    model = make_model(len(features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00005)
    best_error = math.inf
    print(f"Training on {split:,} images, validating on {len(indices) - split:,} ({device})", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, target, mask in train:
            batch = batch.to(device=device, dtype=torch.float32) / 255
            target, mask = target.to(device), mask.to(device)
            flip = (torch.rand(len(batch), device=device) < 0.5).unsqueeze(1)
            batch = torch.where(flip[:, :, None, None], batch.flip(-1), batch)
            mirrored = torch.where(x_axis, -target[:, swap] - 1 / 48, target[:, swap])
            target = torch.where(flip, mirrored, target)
            mask = torch.where(flip, mask[:, swap], mask)
            optimizer.zero_grad()
            loss = squared_error(model(batch), target, mask) / mask.sum().clamp_min(1)
            loss.backward()
            optimizer.step()
        schedule.step()

        model.eval()
        error = torch.zeros((), device=device)
        count = 0
        with torch.inference_mode():
            for batch, target, mask in val:
                batch = batch.to(device=device, dtype=torch.float32) / 255
                error += squared_error(model(batch), target.to(device), mask.to(device))
                count += int(mask.sum())
        rmse = 48 * math.sqrt(error.item() / count)
        print(f"Epoch {epoch:02d}  validation RMSE {rmse:.3f} pixels", flush=True)
        if rmse < best_error:
            best_error = rmse
            best_weights = {name: weight.detach().cpu().clone()
                            for name, weight in model.state_dict().items()}

    model.load_state_dict(best_weights)
    test_images, _, image_ids = load_faces(args.data_dir / "test.csv")
    predictions = []
    with torch.inference_mode():
        for batch in test_images.split(128):
            batch = batch.to(device=device, dtype=torch.float32) / 255
            predictions.append(((model(batch) * 48 + 48).clamp(0, 96)).cpu().numpy())
    by_image = dict(zip(image_ids, np.concatenate(predictions), strict=True))
    by_feature = {name: index for index, name in enumerate(features)}
    count = write_submission(
        args.data_dir, args.submission,
        lambda image_id, feature: float(by_image[image_id][by_feature[feature]]),
    )
    print(f"Best validation RMSE: {best_error:.3f} pixels")
    print(f"Wrote {count:,} predictions to {args.submission}")


if __name__ == "__main__":
    main()
