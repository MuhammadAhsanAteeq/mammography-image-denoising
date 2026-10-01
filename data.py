"""Explicit manifests prevent accidental pairing and patient-split errors."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image


def read_manifest(path):
    path = Path(path).resolve()
    splits = {key: [] for key in ('train', 'val', 'test')}
    patients, seen = {}, set()
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'path', 'patient_id', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest requires path,patient_id,split columns')
        for row in reader:
            split, patient = row['split'].strip(), row['patient_id'].strip()
            if split not in splits or not patient or not row['path'].strip():
                raise ValueError('Each row needs a patient_id, path and train/val/test split')
            image = (path.parent / row['path']).resolve()
            if image in seen:
                raise ValueError('Duplicate image path in manifest')
            if patient in patients and patients[patient] != split:
                raise ValueError('Patient overlap across splits')
            if not image.is_file():
                raise FileNotFoundError(image)
            if image.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                raise ValueError('Use preprocessed 8-bit image files; DICOM is not supported')
            seen.add(image)
            patients[patient] = split
            splits[split].append(image)
    if any(not paths for paths in splits.values()):
        raise ValueError('Provide non-empty train, val and test splits')
    return splits


def load_image(path, size=256):
    if size <= 0 or size % 4:
        raise ValueError('Image size must be a positive multiple of four')
    with Image.open(path) as image:
        if image.mode not in ('L', 'RGB', 'RGBA'):
            raise ValueError('Convert to 8-bit grayscale externally with a documented pipeline')
        image = image.convert('L').resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)[..., None] / 255.0
