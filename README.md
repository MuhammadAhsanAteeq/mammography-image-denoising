# Mammography Image Denoising with Convolutional Autoencoders

**Muhammad Ahsan Ateeq · MSc Artificial Intelligence & Data Science · University of Hull**

An academic computer-vision project investigating whether a compact convolutional denoising autoencoder (CDAE) can reconstruct mammograms corrupted by synthetic Gaussian noise.

**Recorded denoising result: 18.83 dB PSNR · 0.01309 MSE · 28,353 model parameters.** These are historical notebook outputs, not newly reproduced measurements. This is image-reconstruction research, with no clinical deployment or diagnostic validation.

[Portfolio](https://ahsan-career-portfolio.ahsanateeq.chatgpt.site) · [Author](https://github.com/MuhammadAhsanAteeq) · [Methodology and limitations](docs/methodology.md)

## Project overview

The MSc dissertation, *Enhancing Low Radiation Medical Images through Convolutional De-noising Auto-encoders for Improved Image Quality*, explored a two-stage approach:

1. Train an autoencoder to reconstruct clean mammography images.
2. Fine-tune it to map synthetically corrupted images back to clean references.
3. Inspect reconstructions and quantify pixel-level error with PSNR and MSE.

The notebook combines image preprocessing, a custom batch generator, a TensorFlow/Keras encoder-decoder, checkpointing and evaluation. The companion code reorganizes these components into reusable modules with explicit inputs and held-out evaluation.

## Recorded results

| Experiment | PSNR (dB, higher is better) | MSE (lower is better) |
|---|---:|---:|
| Noisy-input denoising | 18.831350 | 0.01308775 |
| Clean-input reconstruction | 31.630733 | 0.00068695 |

Source: uploaded notebook cells 32/34 and 41/43 (zero-based). [Exact values and provenance](results/archived_metrics.json). These experiments have different inputs; clean reconstruction is not a noisy-input baseline, and the table does not quantify denoising improvement over unprocessed noisy images.

The saved arrays show **43,776 training images and 10,930 test images** at **256 × 256 × 1**. The thesis describes approximately 54,700 RSNA mammograms and an 80/20 split. Original images, split membership and checkpoints were not provided, so the metrics and patient separation have not been independently reproduced.

![Archived PSNR and MSE by test image](docs/figures/denoising-metrics.png)

*Sample output preserved from the historical notebook, cell 36. Image index is not a patient count. This figure was not regenerated from raw predictions.*

## Architecture

| Stage | Operations | Output shape |
|---|---|---|
| Input | Normalized grayscale image | 256 × 256 × 1 |
| Encoder 1 | Conv2D, 32 filters, 3 × 3, ReLU; max pool | 128 × 128 × 32 |
| Encoder 2 | Conv2D, 32 filters, 3 × 3, ReLU; max pool | 64 × 64 × 32 |
| Decoder | Two Conv2DTranspose layers, 32 filters, stride 2 | 256 × 256 × 32 |
| Output | Conv2D, 1 filter, 3 × 3, sigmoid | 256 × 256 × 1 |

Optimizer: Adam. Training loss: binary cross-entropy, preserved from the original experiment (distinct from evaluation MSE). The original code requests 50 clean-training epochs and 100 denoising epochs; its saved denoising log stops partway through epoch 11, so 100 completed epochs are not claimed.

## Repository guide

- `notebooks/archived_experiment.ipynb`: sanitized historical code and selected saved outputs; known original defects remain. Read as an archive.
- `notebooks/reproduce.ipynb`: guided companion for a new, reproducible run.
- `denoising/`: image loading, noise, architecture, training and evaluation modules.
- `results/archived_metrics.json`: exact original scalar outputs.
- `docs/`: method notes, source hashes and aggregate result figures.
- `tests/`: metric, noise and patient-split checks.
- `manifest.example.csv`: illustrative input format; contains no real patient records.

## Installation

Use Python 3.11 in a fresh environment. The pinned TensorFlow version is a proposed companion environment, not a recovered original lockfile.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For platform-specific GPU setup, follow the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip). CPU execution is sufficient for a small setup check; full training requires substantial time and memory.

## Dataset setup

The original project uses the [RSNA Screening Mammography Breast Cancer Detection dataset](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data). Obtain access directly and follow its terms. No dataset images, patient metadata, weights or full thesis are redistributed here.

Prepare **8-bit** PNG/JPEG/TIFF grayscale images outside this repository. DICOM windowing/conversion is not implemented; record that preprocessing because it affects results. Create `manifest.csv` using the example columns `path,patient_id,split`; paths are relative to the manifest. Assign each patient wholly to `train`, `val` or `test`, and provide all three splits. The loader rejects patient overlap and duplicate paths. It cannot detect copied images under different names or incorrect patient IDs.

## Train and evaluate

Run from the repository root. Start with one epoch per phase to check your setup:

```bash
python -m denoising.train --manifest manifest.csv --output runs/first-run --pretrain-epochs 1 --epochs 1
python -m denoising.evaluate --manifest manifest.csv --model runs/first-run/denoise.keras --output runs/first-evaluation
```

For an extended experiment, use `--pretrain-epochs 50 --epochs 100`. Default image size is 256, batch size 16, training seed 42 and noise sigma 0.4. Change these through CLI flags. Output folders must be new, preventing accidental replacement of an earlier run.

Training saves phase checkpoints, configuration and loss history. Evaluation saves `metrics.json`, `per_image.csv`, and a clean/noisy/reconstruction `sample.png` from your own held-out data. It reports both the noisy-input baseline and model reconstruction with explicit data range 1.0. Raw images and generated runs are ignored by Git.

Alternatively, open `notebooks/reproduce.ipynb` in a Jupyter-capable editor using the installed environment. Run all cells in order.

## Reproducibility status

Core Python checks pass for known PSNR/MSE values, seeded per-pixel noise, image shape, and patient-overlap rejection. Python files compile, notebook JSON structure and source provenance were checked, and archived aggregate figures were visually inspected.

**Not executed:** TensorFlow training/inference and the companion notebook. The preparation environment lacked TensorFlow/Jupyter; package installation was blocked by network access, and original images/checkpoints were not supplied. After installation and data setup, run the commands above and all notebook cells to complete validation. New workflow results must be reported separately from archived scores.

## Interpretation and next steps

Synthetic noise with standard deviation 0.4 is not evidence of a 40% radiation-dose reduction. Pixel-level similarity does not establish preservation of subtle lesions or diagnostic utility. The original unsorted directory reads leave clean/noisy pairing uncertain, and patient-level separation is unverified. See [methodology](docs/methodology.md) for the complete distinction between the historical study and the companion implementation.

Useful extensions: compare against the noisy-input baseline and classical denoisers, evaluate several noise strengths and seeds, add SSIM with explicit settings, and assess whether reconstruction removes clinically relevant detail using an appropriate expert-led study.

## References and attribution

- Ateeq, Muhammad Ahsan. MSc research notebook and dissertation, University of Hull, 2024. Source-file hashes are retained in `docs/source_manifest.json`.
- RSNA Screening Mammography Breast Cancer Detection dataset, linked above.
- [Keras convolutional autoencoder example](https://keras.io/examples/vision/autoencoder/): a closely matching reference architecture. The supplied notebook does not establish its exact code provenance; this link documents the architectural reference without claiming a novel network design.

No open-source license has been selected yet. Dataset and third-party material remain subject to their own terms.
