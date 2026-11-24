# 👁️ Facial Keypoint Detection

A Python project for the [Kaggle Facial Keypoints Detection competition](https://www.kaggle.com/competitions/facial-keypoints-detection), predicting 15 facial landmarks from 96 × 96 grayscale images.

### 📋 Aims of App

- Inspect a training image with its labeled eye, eyebrow, nose and mouth landmarks
- Compare a mean-landmark baseline, a CNN that predicts coordinates directly and a CNN that predicts a heatmap for each landmark
- Generate a CSV with 27,124 coordinate predictions in Kaggle's `RowId,Location` format

### 👩‍💻 Technical Details

- Python and NumPy for loading image pixels and landmark coordinates
- Pillow for the labeled image preview
- PyTorch for training CNNs on CPU or an available Apple GPU
- A fixed 80/20 split by image for local validation. The loss uses available labels, and horizontal flips mirror the x-coordinates and swap the left and right landmarks
- The heatmap model keeps spatial features and converts each landmark's heatmap into a fractional pixel coordinate

The heatmap submission scored **2.32222 public RMSE** and **2.17159 private RMSE** on Kaggle. Its local validation RMSE was **2.527 pixels**.

The project follows [Daniel Nouri's facial keypoint detection tutorial](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

### 🖥️ Run the app

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pillow torch
```

Download the [competition data from Kaggle](https://www.kaggle.com/competitions/facial-keypoints-detection/data). Extract `training.csv` from `training.zip` and `test.csv` from `test.zip`, then put both CSVs in `data/` along with `IdLookupTable.csv` and `SampleSubmission.csv`.

Inspect the training data and generate a baseline submission.

```bash
python inspect_data.py data landmarks.png
python baseline.py data --submission submission.csv
```

Train the heatmap model and generate its submission.

```bash
python train_landmarks.py data --mode heatmap --epochs 30 --submission submission-heatmap.csv
```

The training script prints validation RMSE by epoch and by landmark for the best epoch, then uses that epoch's model to generate the CSV. Upload `submission-heatmap.csv` on the [Kaggle competition page](https://www.kaggle.com/competitions/facial-keypoints-detection).

For later runs, activate the existing environment and rerun the training command.

```bash
source .venv/bin/activate
python train_landmarks.py data --mode heatmap --epochs 30 --submission submission-heatmap.csv
```

### 💭 Future Improvements

- Test whether grayscale contrast normalization helps on sharp, low-contrast images and measure its effect on blurrier images.
