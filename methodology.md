# Methodology, provenance and limitations

## Historical experiment

The submitted notebook is the authority for the precise stored metrics: denoising PSNR 18.831350149874776 dB and MSE 0.013087749832030362; clean reconstruction PSNR 31.630733212845243 dB and MSE 0.00068695245307357. Both pairs satisfy PSNR = -10 log10(MSE) with data range 1. They are global array metrics, not the arithmetic mean of per-image PSNR.

The thesis's abstract, results discussion and conclusion use differing denoising PSNR figures (including 18.31, 18.831, 18.13 and 18.5). This repository uses the exact saved notebook outputs. Thesis page 12 shows the same MSE output as the notebook. Its approximate 54,700-image description differs slightly from the saved array counts totaling 54,706; neither is silently substituted for the other.

The thesis describes DICOM-to-PNG conversion, resizing to 256 × 256, and an 80/20 training/test split. The conversion and split-building code are absent. Original files, model weights, predictions and patient-level split assignments were not uploaded. Per-image membership, pairing and results therefore cannot be independently verified.

## Original execution issues

- Local Windows paths prevent portable execution.
- Cell 10 calls a two-argument noise-saving function with only one argument and passes arrays where filenames are expected.
- The unused `noise()` helper draws a single scalar Gaussian value rather than an array. The saving function draws per-pixel noise at 512 × 512 before later resizing to 256 × 256.
- Independent unsorted directory listings load clean/noisy test arrays; matching order is not enforced.
- The custom generator emits arrays without an explicit final channel dimension.
- Checkpoint `period` is a legacy argument. Denoising training monitors `val_loss` without supplying validation data; the stored log confirms skipped checkpoint saves.
- A denoising checkpoint is loaded under a different filename from the clean-stage save, leaving its lineage uncertain.
- Clean training uses `test_data` for validation. The historical test set therefore is not established as an untouched final evaluation set.
- A 100-epoch denoising call exists, but saved logs stop during epoch 11. Completion of all 100 epochs is not established.

The archived notebook retains these issues explicitly. Its runtime warnings, student identifier, local machine paths and sample mammograms were removed for a focused public artifact. Scalar outputs, model summary and aggregate metric plots were retained. The original uploads remain unchanged.

## Companion implementation

The refactor preserves the recorded convolutional architecture, Adam optimizer, binary cross-entropy loss and clean-then-noisy training concept. It adds explicit patient-separated manifests, paired in-memory noise generation, consistent grayscale shape, configurable paths, seeded randomness, streaming batches, validation checkpoints and held-out evaluation. The default batch size is reduced from 128 to 16 to make setup more approachable.

This is **a new experimental implementation**, not an exact reproduction. Noise is now applied after resizing as zero-mean Gaussian noise with sigma 0.4 on the [0,1] scale, clipped to that range. Original noise was generated on saved 512 × 512 images before resizing, so effective noise characteristics differ. The new validation split, checkpoint selection and random sequence also differ. Random seeds do not promise bitwise reproducibility across devices.

## Evaluation

MSE is the mean of squared differences for normalized clean and reconstructed pixels. Global PSNR uses data range 1.0 and the global MSE. The companion additionally reports mean per-image PSNR and the unprocessed noisy-image baseline on the same held-out samples. It does not substitute the archived clean-input reconstruction for that baseline.

Tests verify metrics against a known numerical example, per-pixel seeded noise, accepted image dimensions, and patient-overlap rejection. Full training, checkpoint loading and notebook execution remain unvalidated until the required packages and authorized dataset are available. The repository does not include pretrained weights.

## Research scope

The experiment explores synthetic corruption, not actual low-dose acquisitions. The historical phrase “40% noise” refers to a Gaussian scale factor of 0.4; it does not establish a radiation dose, corruption percentage or clinical performance. PSNR and MSE cannot alone establish diagnostic safety, lesion preservation or clinical benefit.
