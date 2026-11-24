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


from train_cnn import flip_indices, load_faces, squared_error


class LandmarkModel(nn.Module):
    def __init__(self, mode, outputs):
        super().__init__()
        self.mode = mode
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU())
        self.pool = nn.AvgPool2d(2)
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        if mode == "heatmap":
            self.head = nn.Sequential(
                nn.Conv2d(80, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, outputs // 2, 1),
            )
            axis = (torch.arange(24, dtype=torch.float32) * 4 + 1.5 - 48) / 48
            y, x = torch.meshgrid(axis, axis, indexing="ij")
            self.register_buffer("grid", torch.stack((x.flatten(), y.flatten()), dim=-1))
        else:
            self.head = nn.Sequential(
                nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 6 * 6, 128),
                nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, outputs),
            )

    def forward(self, images):
        if self.mode == "control":
            fine = nn.functional.max_pool2d(self.stem(self.pool(images)), 2)
        else:
            fine = nn.functional.max_pool2d(self.pool(self.stem(images)), 2)
        coarse = self.encoder(fine)
        if self.mode != "heatmap":
            return self.head(coarse), None
        context = nn.functional.interpolate(coarse, size=fine.shape[-2:],
                                            mode="bilinear", align_corners=False)
        logits = self.head(torch.cat((fine, context), dim=1)).flatten(2)
        log_probs = logits.log_softmax(dim=-1)
        coordinates = (log_probs.exp() @ self.grid).flatten(1)
        return coordinates, log_probs



def main():
    parser = argparse.ArgumentParser(description="Train a CNN on available facial coordinates.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "late-pool", "heatmap"), required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

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
    model = LandmarkModel(args.mode, len(features)).to(device)
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
            coordinates, log_probs = model(batch)
            loss = squared_error(coordinates, target, mask) / mask.sum().clamp_min(1)
            if log_probs is not None:
                pairs = target.reshape(len(batch), -1, 2)
                labeled_pairs = mask.reshape(len(batch), -1, 2).all(dim=-1)
                distance = (pairs.unsqueeze(2) - model.grid).square().sum(dim=-1)
                target_maps = (-distance / (2 * (4 / 48) ** 2)).softmax(dim=-1)
                heatmap_loss = (-(target_maps * log_probs).sum(dim=-1) * labeled_pairs).sum()
                loss = loss + 0.01 * heatmap_loss / labeled_pairs.sum().clamp_min(1)
            loss.backward()
            optimizer.step()
        schedule.step()

        model.eval()
        error = torch.zeros(len(features), device=device)
        count = torch.zeros(len(features), dtype=torch.long, device=device)
        with torch.inference_mode():
            for batch, target, mask in val:
                batch = batch.to(device=device, dtype=torch.float32) / 255
                coordinates, _ = model(batch)
                mask = mask.to(device)
                error += ((coordinates - target.to(device)).square() * mask).sum(dim=0)
                count += mask.sum(dim=0)
        rmse = 48 * math.sqrt(error.sum().item() / count.sum().item())
        print(f"Epoch {epoch:02d}  validation RMSE {rmse:.3f} pixels", flush=True)
        if rmse < best_error:
            best_error = rmse
            best_by_landmark = 48 * (error.reshape(-1, 2).sum(1) / count.reshape(-1, 2).sum(1)).sqrt().cpu()
            best_weights = {name: weight.detach().cpu().clone()
                            for name, weight in model.state_dict().items()}

    model.load_state_dict(best_weights)
    test_images, _, image_ids = load_faces(args.data_dir / "test.csv")
    predictions = []
    with torch.inference_mode():
        for batch in test_images.split(128):
            batch = batch.to(device=device, dtype=torch.float32) / 255
            coordinates, _ = model(batch)
            predictions.append((coordinates * 48 + 48).clamp(0, 96).cpu().numpy())
    by_image = dict(zip(image_ids, np.concatenate(predictions), strict=True))
    by_feature = {name: index for index, name in enumerate(features)}
    count = write_submission(
        args.data_dir, args.submission,
        lambda image_id, feature: float(by_image[image_id][by_feature[feature]]),
    )
    print(f"Best validation RMSE: {best_error:.3f} pixels")
    for name, score in zip(features[::2], best_by_landmark, strict=True):
        print(f"{name[:-2]}: {score:.3f} pixels")
    print(f"Wrote {count:,} predictions to {args.submission}")


if __name__ == "__main__":
    main()
