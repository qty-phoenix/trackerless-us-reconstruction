import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import torch
from geometry import (ScanPrediction, image_points, landmark_ddfs, score_scan,
                      global_from_labels, local_from_global)
from inference import stitch_windows
from datasets.longterm import LongTermDataset
from baselines.nr_rec_fus.datasets.nr_rec_dataset import NRRecDataset
from baselines.nr_rec_fus.models.rigid_pose import se3_from_vec
from baselines.nr_rec_fus.models.nr_rec_fus import NRRecFUS
from baselines.nr_rec_fus.models.volume import splat, volume_bounds, bending_energy
from baselines.nr_rec_fus.train import loss_and_metrics
from models_longterm import LongTermEfficientNet, relative_tool_transforms
from training import run_training, seed_all


class GeometryTests(unittest.TestCase):
    def test_official_pixels_calibration_landmark_index_and_direction(self):
        points = image_points(2, 3)
        torch.testing.assert_close(points[:2], torch.tensor([[1., 2., 3., 1., 2., 3.], [1., 1., 1., 2., 2., 2.]], dtype=torch.float64))
        # Nontrivial spatial calibration: image x maps to tool y.
        c = torch.tensor([[0., -1., 0., 7.], [1., 0., 0., -3.], [0., 0., 1., 2.], [0., 0., 0., 1.]], dtype=torch.float64)
        s = torch.diag(torch.tensor([.2, .3, 1., 1.], dtype=torch.float64))
        g = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        g[1, 1, 3], g[2, 1, 3] = 2, 5
        prediction = ScanPrediction(g, s, c)
        gl, ll = landmark_ddfs(prediction, np.array([[1, 2, 1], [2, 3, 2]]))
        torch.testing.assert_close(gl, torch.tensor([[2., 5.], [0., 0.], [0., 0.]], dtype=torch.float64))
        torch.testing.assert_close(ll, torch.tensor([[2., 3.], [0., 0.], [0., 0.]], dtype=torch.float64))
        # Identity ground truth: mean global displacement 3.5, local 2.5 mm.
        metrics = score_scan(prediction, torch.eye(4).repeat(3, 1, 1), [[1, 2, 1], [2, 3, 2]], 2, 3, 2)
        self.assertEqual(metrics, {'GPE': 3.5, 'GLE': 3.5, 'LPE': 2.5, 'LLE': 2.5})

    def test_noncommuting_rotations_and_chunk_invariance(self):
        params = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 2., 3., .3, -.2, .1], [-2., 1., 4., -.1, .4, .2]], dtype=torch.float64)
        labels = se3_from_vec(params)
        global_t = global_from_labels(labels)
        torch.testing.assert_close(local_from_global(global_t), torch.linalg.inv(labels[:-1]) @ labels[1:])
        prediction = ScanPrediction(global_t, torch.eye(4, dtype=torch.float64), labels[1])
        a = score_scan(prediction, labels, [[1, 2, 1], [2, 3, 2]], 2, 3, 1)
        b = score_scan(prediction, labels, [[1, 2, 1], [2, 3, 2]], 2, 3, 20)
        self.assertEqual(a, b)
        self.assertTrue(all(v == 0 for v in a.values()))
        with self.assertRaises(ValueError):
            landmark_ddfs(prediction, [[0, 1, 1]])
        with self.assertRaises(ValueError):
            landmark_ddfs(prediction, [[3, 1, 1]])

    def test_stitch_tail_short_and_rotations(self):
        for n in (2, 4, 5, 7, 8, 11):
            params = torch.zeros(n, 6, dtype=torch.float64)
            params[:, 0] = torch.arange(n)
            params[:, 5] = torch.arange(n) * .12
            expected = se3_from_vec(params)
            calls = []
            def predict(start, stop):
                calls.append((start, stop))
                return torch.linalg.inv(expected[start]) @ expected[start:stop]
            actual = stitch_windows(n, 4, predict)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(calls[-1][1], n)

    def test_nonrigid_identity_anchor_and_zero_field_reduction(self):
        eye = torch.eye(4, dtype=torch.float64)
        g = eye.repeat(3, 1, 1)
        g[1, 0, 3], g[2, 0, 3] = 1, 2
        bounds = (torch.tensor([[-10., -10., -10.]]), torch.tensor([[10., 10., 10.]]))
        field = torch.zeros(1, 3, 8, 8, 8)
        p = image_points(2, 3)
        rigid = ScanPrediction(g, eye, eye)
        refined = ScanPrediction(g, eye, eye, field, bounds)
        torch.testing.assert_close(refined.positions(0, p), p[:3])
        for local in (False, True):
            torch.testing.assert_close(refined.displacement(2, p, local), rigid.displacement(2, p, local))
        # d_x(x)=0.1*x: first plane stays fixed; x translation grows 10%.
        field[:, 0] = torch.linspace(-1, 1, 8)[None, None, :]
        torch.testing.assert_close(refined.positions(0, p), p[:3])
        expected = torch.zeros(3, 6, dtype=torch.float64)
        expected[0] = 2.2
        torch.testing.assert_close(refined.displacement(2, p), expected, atol=1e-6, rtol=1e-6)
        for chunk in (1, 6):
            metrics = score_scan(refined, eye.repeat(3, 1, 1), [[1, 1, 1], [2, 3, 2]], 2, 3, chunk)
            for key in ('GPE', 'GLE'):
                self.assertAlmostEqual(metrics[key], 1.65, places=6)
            for key in ('LPE', 'LLE'):
                self.assertAlmostEqual(metrics[key], 1.1, places=6)


class TrainingTests(unittest.TestCase):
    def test_split_file_loader_fixed_validation_and_epoch_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with h5py.File(root / 'frames.h5', 'w') as f:
                f['frames'] = np.arange(30 * 8 * 8, dtype=np.uint8).reshape(30, 8, 8)
            with h5py.File(root / 'poses.h5', 'w') as f:
                f['tforms'] = np.tile(np.eye(4), (30, 1, 1))
            row = {'frames_path': 'frames.h5', 'tforms_path': 'poses.h5', 'id': '050/scan', 'subject': '050'}
            for split in ('train', 'val'):
                (root / f'{split}.jsonl').write_text(json.dumps(row) + '\n')
            for cls in (NRRecDataset, LongTermDataset):
                ds = cls(root, 'val', root, 4, 8, size=(8, 8))
                ids = [ds[i]['indices'] for i in range(len(ds))]
                ds.set_epoch(19)
                for i in range(len(ds)):
                    torch.testing.assert_close(ids[i], ds[i]['indices'])
                    self.assertEqual(ds[i]['tforms'].shape, (4, 4, 4))
                self.assertEqual(len(ds._files), 2)
                ds.close()

    def test_rodrigues_zero_has_rotation_gradient(self):
        params = torch.zeros(1, 6, requires_grad=True)
        rotation = se3_from_vec(params)
        torch.testing.assert_close(rotation[0], torch.eye(4))
        rotation[0, 0, 1].backward()
        self.assertAlmostEqual(params.grad[0, 5].item(), -1.)
        self.assertTrue(torch.isfinite(params.grad).all())

    def test_longterm_configured_frame_count(self):
        model = LongTermEfficientNet(num_samples=4).eval()
        with torch.no_grad():
            self.assertEqual(model(torch.rand(1, 4, 32, 32)).shape, (1, 6, 6))
        self.assertEqual(relative_tool_transforms(torch.eye(4).repeat(1, 4, 1, 1)).shape, (1, 6, 4, 4))

    def test_trilinear_splat_conserves_mass_and_point_gradient(self):
        points = torch.tensor([[[[.25], [.5], [.75]]]], requires_grad=True)
        bounds = (torch.zeros(1, 3), torch.ones(1, 3))
        sums, weights = splat(points, torch.ones(1, 1, 1, 1) * .7, bounds, (4, 4, 4), True)
        self.assertAlmostEqual(weights.sum().item(), 1.)
        self.assertAlmostEqual(sums.sum().item(), .7, places=6)
        (sums * torch.arange(4)[None, None, None, None, :]).sum().backward()
        self.assertGreater(points.grad.abs().sum().item(), 0)

    def test_nr_geometry_and_registration_are_coupled(self):
        torch.manual_seed(0)
        model = NRRecFUS(4, volume_shape=(8, 8, 8), deform_channels=2).eval()
        frames = torch.rand(1, 4, 32, 32)
        scale = torch.diag(torch.tensor([.2, .3, 1., 1.]))
        t = torch.eye(4).repeat(1, 4, 1, 1)
        t[0, :, 2, 3] = torch.arange(4)
        config = {'reg_loss_weight': 1000., 'smooth_weight': .01, 'refined_loss_weight': 1.}
        loss, metrics = loss_and_metrics(model, {'frames': frames, 'tforms': t}, scale, torch.eye(4), config, 'cpu')
        rigid_grad = torch.autograd.grad(metrics['registration_loss'], tuple(model.rigid.parameters()), retain_graph=True, allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in rigid_grad if g is not None), 0)
        deform_grad = torch.autograd.grad(metrics['refined_loss'], tuple(model.deform.parameters()), retain_graph=True, allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in deform_grad if g is not None), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, subject):
                self.rows = [{'subject': subject}]
            def __len__(self): return 3
            def __getitem__(self, i): return {'frames': torch.tensor([float(i)])}
            def set_epoch(self, epoch): pass
            def close(self): pass
        def objective(model, batch):
            error = model(batch['frames']) - 2
            return error.square().mean(), {'distance_mm': error.abs().mean()}
        def run(path, epochs, resume=''):
            seed_all(7)
            model = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
            run_training(model, TinyDataset('000'), TinyDataset('050'), objective,
                         {'seed': 7, 'method': 'tiny'}, 'cpu', path, epochs, 2, .01, resume=resume)
            return torch.load(path / 'last.pt', weights_only=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = run(root / 'full', 2)
            run(root / 'resumed', 1)
            resumed = run(root / 'resumed', 2, str(root / 'resumed/last.pt'))
            self.assertEqual(full['history'], resumed['history'])
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
            self.assertTrue((root / 'resumed/best.pt').exists())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
