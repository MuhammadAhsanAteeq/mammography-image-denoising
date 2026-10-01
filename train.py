"""Run clean reconstruction pretraining followed by noisy-to-clean fine-tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .data import read_manifest, load_image
from .metrics import add_noise
from .model import build_model


class ImageBatches(tf.keras.utils.Sequence):
    def __init__(self, paths, size, batch_size, sigma, seed, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        self.paths, self.size, self.batch_size = paths, size, batch_size
        self.sigma, self.seed, self.shuffle = sigma, seed, shuffle
        self.epoch = 0
        self.order = np.arange(len(paths))
        if shuffle:
            np.random.default_rng(seed).shuffle(self.order)

    def __len__(self):
        return (len(self.paths) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        indices = self.order[index*self.batch_size:(index+1)*self.batch_size]
        clean = np.stack([load_image(self.paths[i], self.size) for i in indices])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        return add_noise(clean, self.sigma, rng), clean

    def on_epoch_end(self):
        if self.shuffle:
            self.epoch += 1
            np.random.default_rng(self.seed + self.epoch).shuffle(self.order)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', default='runs/experiment')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--pretrain-epochs', type=int, default=50)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--sigma', type=float, default=0.4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.batch_size < 1 or args.pretrain_epochs < 0 or args.epochs < 1 or args.sigma < 0:
        p.error('Batch size/epochs must be positive; pretraining/sigma may be zero')
    data = read_manifest(args.manifest)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # prevent overwriting an earlier experiment
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(args.size)
    history = {}
    for phase, sigma, epochs in [('pretrain', 0.0, args.pretrain_epochs), ('denoise', args.sigma, args.epochs)]:
        if epochs == 0:
            continue
        train = ImageBatches(data['train'], args.size, args.batch_size, sigma, args.seed, shuffle=True)
        val = ImageBatches(data['val'], args.size, args.batch_size, sigma, args.seed + 1)
        checkpoint = out / f'{phase}.keras'
        callback = tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_loss', save_best_only=True)
        history[phase] = model.fit(train, validation_data=val, epochs=epochs, callbacks=[callback]).history
        model = tf.keras.models.load_model(checkpoint)
    metadata = vars(args) | {'tensorflow': tf.__version__, 'numpy': np.__version__,
        'counts': {s:len(paths) for s, paths in data.items()},
        'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'note':'Refactored run; not a reproduction of archived scalar results'}
    (out/'config.json').write_text(json.dumps(metadata, indent=2))
    (out/'history.json').write_text(json.dumps(history, indent=2))
    print(f'Saved model: {out / "denoise.keras"}')


if __name__ == '__main__':
    main()
