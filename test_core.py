import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from denoising.data import read_manifest, load_image
from denoising.metrics import quality, add_noise


class CoreTests(unittest.TestCase):
    def test_known_metrics(self):
        result = quality(np.zeros((2,2)), np.full((2,2), .1))
        self.assertAlmostEqual(result['mse'], .01)
        self.assertAlmostEqual(result['psnr_db'], 20)
        self.assertTrue(np.isinf(quality(np.zeros(1), np.zeros(1))['psnr_db']))

    def test_noise_is_seeded_and_per_pixel(self):
        x = np.full((64,64,1), .5, dtype=np.float32)
        a = add_noise(x, .1, np.random.default_rng(42))
        b = add_noise(x, .1, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.std(a-x), .09)
        self.assertLess(np.std(a-x), .11)
        self.assertTrue(((a >= 0) & (a <= 1)).all())

    def test_manifest_patient_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(3): Image.fromarray(np.full((8,8), 128, dtype=np.uint8)).save(root/f'{i}.png')
            path = root/'manifest.csv'
            def write(patients):
                with path.open('w', newline='') as f:
                    w = csv.writer(f); w.writerow(['path','patient_id','split'])
                    w.writerows((f'{i}.png', patients[i], s) for i,s in enumerate(['train','val','test']))
            write(['a','b','c'])
            self.assertEqual(len(read_manifest(path)['test']), 1)
            self.assertEqual(load_image(root/'0.png', 16).shape, (16,16,1))
            write(['a','a','c'])
            with self.assertRaisesRegex(ValueError, 'Patient overlap'): read_manifest(path)


if __name__ == '__main__': unittest.main()
