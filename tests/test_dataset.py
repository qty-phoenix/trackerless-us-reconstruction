import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np

from datasets import TUSREC2024


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        with h5py.File(self.root / 'sample.h5', 'w') as h:
            h['frames'] = np.arange(30, dtype=np.uint8).reshape(5, 2, 3)
            h['tforms'] = np.tile(np.eye(4), (5, 1, 1))
            h['landmark'] = np.array([[1, 1, 1]])
        (self.root / 'calib_matrix.csv').write_text(
            'scaling_from_pixel_to_mm\n1,0,0,0\n0,1,0,0\n0,0,1,0\n0,0,0,1\n'
            'spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system\n'
            '1,0,0,0\n0,1,0,0\n0,0,1,0\n0,0,0,1\n')
        (self.root / 'manifests').mkdir()
        rows = [{'id': f'050/scan{i}', 'subject': '050', 'num_frames': 5,
                 'frames_path': 'sample.h5', 'tforms_path': 'sample.h5',
                 'landmarks_path': 'sample.h5', 'landmarks_key': 'landmark'} for i in range(72)]
        (self.root / 'manifests/val.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))

    def tearDown(self):
        self.tmp.cleanup()

    def test_inference_does_not_return_pose_labels(self):
        sample = TUSREC2024(self.root, 'val', window=3)[0]
        self.assertNotIn('targets', sample)
        self.assertEqual(sample['frames'].dtype, np.uint8)
        sample['calibration']['pixel_to_mm'][0, 0] = 9
        self.assertEqual(TUSREC2024(self.root, 'val')[0]['calibration']['pixel_to_mm'][0, 0], 1)

    def test_windows_include_tail_without_crossing_scans(self):
        ds = TUSREC2024(self.root, 'val', window=3, stride=3, include_targets=True)
        self.assertEqual(len(ds), 144)
        np.testing.assert_array_equal(ds[1]['frame_indices'], [2, 3, 4])
        self.assertEqual(ds[2]['scan_id'], '050/scan1')
        np.testing.assert_array_equal(ds[2]['frame_indices'], [0, 1, 2])
        self.assertEqual(ds[0]['targets']['tool_to_world'].shape, (3, 4, 4))

    def test_incomplete_split_and_test_split_rejected(self):
        with self.assertRaises(ValueError):
            TUSREC2024(self.root, 'test')
        (self.root / 'manifests/val.jsonl').write_text('')
        with self.assertRaises(ValueError):
            TUSREC2024(self.root, 'val')

if __name__ == '__main__':
    unittest.main()
