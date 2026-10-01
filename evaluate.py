"""Evaluate a saved model on the held-out split, including a noisy-input baseline."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import read_manifest, load_image
from .metrics import add_noise, quality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', default='runs/evaluation')
    p.add_argument('--sigma', type=float, default=0.4)
    p.add_argument('--seed', type=int, default=44)
    args = p.parse_args()
    if args.sigma < 0:
        p.error('sigma must be non-negative')
    paths = read_manifest(args.manifest)['test']
    model = tf.keras.models.load_model(args.model)
    size = model.input_shape[1]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    rows = []
    for index, path in enumerate(paths):
        clean = load_image(path, size)
        noisy = add_noise(clean, args.sigma, rng)
        predicted = np.clip(model(noisy[None], training=False).numpy()[0], 0, 1)
        baseline, denoised = quality(clean, noisy), quality(clean, predicted)
        rows.append({'sample_index':index, 'noisy_mse':baseline['mse'], 'noisy_psnr_db':baseline['psnr_db'],
                     'denoised_mse':denoised['mse'], 'denoised_psnr_db':denoised['psnr_db']})
        if index == 0:
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            for axis, array, title in zip(axes, [clean, noisy, predicted], ['Clean reference', 'Synthetic noise', 'Model reconstruction']):
                axis.imshow(array[..., 0], cmap='gray', vmin=0, vmax=1)
                axis.set_title(title)
                axis.axis('off')
            fig.suptitle('Held-out sample 0 | synthetic Gaussian noise')
            fig.tight_layout()
            fig.savefig(out/'sample.png', dpi=160)
            plt.close(fig)
    with (out/'per_image.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'n_images':len(rows), 'sigma':args.sigma, 'seed':args.seed, 'data_range':1.0}
    for group in ('noisy', 'denoised'):
        mse = float(np.mean([r[f'{group}_mse'] for r in rows]))
        summary[group] = {'global_mse':mse, 'global_psnr_db':float(-10*np.log10(mse)) if mse else 'Infinity',
                         'mean_image_psnr_db':float(np.mean([r[f'{group}_psnr_db'] for r in rows]))}
    # Represent exact reconstruction as a string to retain strict JSON compatibility.
    def finite_json(value):
        if isinstance(value, dict): return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value, float) and not np.isfinite(value): return str(value)
        return value
    (out/'metrics.json').write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False))
    print(json.dumps(finite_json(summary), indent=2))


if __name__ == '__main__':
    main()
