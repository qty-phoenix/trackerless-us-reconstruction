"""Image-only full-scan inference shared by the adapted baselines."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from geometry import (ScanPrediction, calibration_tensors, image_points,
                      iter_pixel_ddfs, landmark_ddfs)
from models_longterm import LongTermEfficientNet, pair_samples, euler_xyz_to_matrix
from training import select_device


def read_window(frames, start, stop, size, device):
    x = torch.as_tensor(np.asarray(frames[start:stop]).copy(), device=device).float() / 255.
    return F.interpolate(x[:, None], size=size, mode='bilinear', align_corners=False).squeeze(1)


def stitch_windows(n, window, predict_window):
    """Anchor each window on an already placed frame; cover short scans and tail.

    predict_window(start, stop) returns K first-frame-relative tool transforms.
    Shared policy: stride K-1, final overlapping window, retain existing frames.
    Long-Term uses its (0,j) predictions, preserving within-window long-range edges.
    """
    if n < 2 or window < 2:
        raise ValueError('Require at least two scan and input frames')
    global_t = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    if n <= window:
        starts = [0]
    else:
        starts = list(range(0, n - window + 1, window - 1))
        if starts[-1] != n - window:
            starts.append(n - window)
    placed = 0
    for start in starts:
        stop = min(start + window, n)
        relative = predict_window(start, stop).detach().cpu().double()
        if relative.shape != (stop - start, 4, 4) or not torch.isfinite(relative).all():
            raise ValueError('Invalid window transforms')
        for index in range(max(start + 1, placed + 1), stop):
            global_t[index] = global_t[start] @ relative[index - start]
        placed = max(placed, stop - 1)
    return global_t


class Predictor:
    def __init__(self, checkpoint, device='cuda', method=None, rigid_only=False,
                 online_epochs=None):
        self.device = select_device(device)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.config = dict(saved.get('config', saved.get('args', {})))
        inferred = saved.get('method', self.config.get('method', 'longterm'))
        if method is not None and method != inferred:
            raise ValueError(f'Checkpoint method {inferred} does not match {method}')
        self.method = inferred
        self.rigid_only = rigid_only
        if online_epochs is not None:
            if self.method != 'recon' or online_epochs < 0:
                raise ValueError('--online-epochs applies only to RecON and must be nonnegative')
            self.config['online_epochs'] = online_epochs
        self.window = self.config.get('num_samples', 10 if self.method == 'longterm' else 4)
        self.size = self.config.get('height', 120), self.config.get('width', 160)
        if self.method == 'longterm':
            self.model = LongTermEfficientNet(num_samples=self.window)
        elif self.method == 'moglo_net':
            from baselines.moglo_net.models import MoGLoNet
            self.model = MoGLoNet(num_samples=self.window)
        elif self.method == 'nr_rec_fus':
            from baselines.nr_rec_fus.train import build_model
            self.model = build_model(self.config)
        elif self.method == 'recon':
            from baselines.recon.models import RecON
            self.model = RecON(imagenet_pretrained=False)
        elif self.method == 'dc2_net':
            from baselines.dc2_net.models import DC2Net
            self.model = DC2Net(tuple(self.config.get('model_layers', (3, 4, 6, 3))),
                                self.config.get('groups', 32),
                                self.config.get('base_width', 4))
        elif self.method == 'two_frame_cnn':
            from baselines.two_frame_cnn.models import TwoFrameCNN
            self.model = TwoFrameCNN(self.config.get('dropout', .25),
                                     self.config.get('height', 128),
                                     self.config.get('width', 128))
        else:
            raise ValueError(f'Unknown method {self.method}')
        self.model.load_state_dict(saved['model'])
        self.model.to(self.device).eval()
        self.online_bank = saved.get('online_bank')
        self.recon_initial_state = saved['model'] if self.method == 'recon' else None

    def predict_scan(self, frames, data_path_calib):
        if len(frames.shape) != 3 or tuple(frames.shape[1:]) != (480, 640) or np.dtype(frames.dtype) != np.uint8:
            raise ValueError('Expected uint8 frames [N,480,640]')
        if self.method == 'recon':
            return self._predict_recon(frames, data_path_calib)
        if self.method == 'dc2_net':
            return self._predict_dc2(frames, data_path_calib)
        if self.method == 'two_frame_cnn':
            return self._predict_two_frame_cnn(frames, data_path_calib)

        def window_prediction(start, stop):
            x = read_window(frames, start, stop, self.size, self.device)
            length = len(x)
            if length < self.window:
                x = torch.cat((x, x[-1:].expand(self.window - length, -1, -1)))
            if self.method == 'longterm':
                transforms = euler_xyz_to_matrix(self.model(x[None]))[0]
                pairs = pair_samples(self.window, self.window - 1)
                relative = transforms[pairs[:, 0] == 0]
            elif self.method == 'moglo_net':
                from baselines.moglo_net.models import se3_from_vec
                scale = torch.as_tensor(self.config['motion_scale'], device=self.device, dtype=x.dtype)
                predictions, _, _ = self.model(x[None, :, None])
                relative = se3_from_vec(predictions.mean(1)[0] * scale)
            else:
                from baselines.nr_rec_fus.models.rigid_pose import se3_from_vec
                relative = se3_from_vec(self.model.rigid(x[None]))[0]
            identity = torch.eye(4, device=self.device)[None]
            return torch.cat((identity, relative))[:length]

        with torch.no_grad():
            global_t = stitch_windows(len(frames), self.window, window_prediction)
            scale, spatial = calibration_tensors(data_path_calib, dtype=torch.float64)
            prediction = ScanPrediction(global_t, scale, spatial)
            if self.method == 'nr_rec_fus' and not self.rigid_only:
                self._refine_scan(prediction, frames, data_path_calib)
        return prediction

    def _predict_two_frame_cnn(self, frames, data_path_calib):
        """Predict every adjacent image transform and compose the scan trajectory."""
        from baselines.recon.models import chain_relative, dof_to_matrix
        n = len(frames)
        if n < 2:
            raise ValueError('A scan needs at least two frames')
        batch_size = self.config.get('inference_batch_size', 64)
        predictions = []
        with torch.no_grad():
            for start in range(0, n - 1, batch_size):
                stop = min(start + batch_size, n - 1)
                value = read_window(frames, start, stop + 1, self.size, self.device)
                # Match the uint8 cache used for formal training exactly.
                value = value.mul(255).round_().div_(255)
                pairs = torch.stack((value[:-1], value[1:]), 1)
                predictions.append(self.model(pairs))
            motion = torch.cat(predictions)
            motion = torch.cat((motion[:, :3], torch.deg2rad(motion[:, 3:])), -1)
            global_image = chain_relative(dof_to_matrix(motion))
            _, spatial = calibration_tensors(data_path_calib, self.device,
                                              dtype=torch.float32)
            tool_global = spatial @ global_image @ torch.linalg.inv(spatial)
            scale64, spatial64 = calibration_tensors(data_path_calib, dtype=torch.float64)
        return ScanPrediction(tool_global.detach().cpu().double(), scale64, spatial64)

    def _predict_dc2(self, frames, data_path_calib):
        """Aggregate every overlapping segment's mean motion on covered edges."""
        from baselines.recon.models import chain_relative, dof_to_matrix
        n = len(frames)
        if n < 2:
            raise ValueError('A scan needs at least two frames')
        window = self.window
        starts = list(range(max(1, n - window + 1)))
        batch_size = self.config.get('inference_batch_size', 16)
        motion_scale = torch.as_tensor(self.config['motion_scale'], device=self.device,
                                       dtype=torch.float32)
        predictions = []
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                group = []
                for start in starts[offset:offset + batch_size]:
                    value = read_window(frames, start, min(start + window, n),
                                        self.size, self.device)
                    if len(value) < window:
                        value = torch.cat((value, value[-1:].expand(window - len(value), -1, -1)))
                    group.append(value)
                estimate, _, _ = self.model(torch.stack(group))
                predictions.append(estimate * motion_scale)
            estimates = torch.cat(predictions)
            edge_sum = estimates.new_zeros(n - 1, 6)
            edge_count = estimates.new_zeros(n - 1, 1)
            for estimate, start in zip(estimates, starts):
                stop = min(start + window - 1, n - 1)
                edge_sum[start:stop] += estimate
                edge_count[start:stop] += 1
            if (edge_count == 0).any():
                raise RuntimeError('DC2-Net sliding windows did not cover every scan edge')
            motion = edge_sum / edge_count
            motion = torch.cat((motion[:, :3], torch.deg2rad(motion[:, 3:])), -1)
            global_image = chain_relative(dof_to_matrix(motion))
            _, spatial = calibration_tensors(data_path_calib, self.device, dtype=torch.float32)
            tool_global = spatial @ global_image @ torch.linalg.inv(spatial)
            scale64, spatial64 = calibration_tensors(data_path_calib, dtype=torch.float64)
        return ScanPrediction(tool_global.detach().cpu().double(), scale64, spatial64)

    def _predict_recon(self, frames, data_path_calib):
        from baselines.recon.models import chain_relative, dof_to_matrix, unscaled_motion
        from baselines.recon.online import OnlineAdapter
        x = read_window(frames, 0, len(frames), self.size, self.device)[None]
        reco_size = self.config['reconstruction_height'], self.config['reconstruction_width']
        reco_scale, spatial = calibration_tensors(data_path_calib, self.device,
                                                   dtype=torch.float32, size=reco_size)
        adapter = OnlineAdapter(self.model, self.config, self.online_bank, self.device, reco_scale,
                                self.recon_initial_state)
        scan_seed = int(torch.as_tensor(np.asarray(frames[0])).sum()) + len(frames) * 1000003
        network_motion = adapter.predict(x, scan_seed)
        relative_image = dof_to_matrix(unscaled_motion(network_motion,
                                                        self.config['angle_scale']))[0]
        global_image = chain_relative(relative_image)
        tool_global = spatial @ global_image @ torch.linalg.inv(spatial)
        scale, spatial64 = calibration_tensors(data_path_calib, dtype=torch.float64)
        return ScanPrediction(tool_global.detach().cpu().double(), scale, spatial64)

    def _refine_scan(self, prediction, frames, data_path_calib):
        # Build one full-scan volume using only predicted geometry. Accumulate
        # sums before division, so temporal chunk size cannot change intensities.
        from baselines.nr_rec_fus.models.volume import volume_bounds, splat
        scale, _ = calibration_tensors(data_path_calib, self.device, size=self.size)
        transforms = prediction.image_global().to(device=self.device, dtype=torch.float32)
        points = scale @ image_points(*self.size, device=self.device, dtype=torch.float32)
        corners = points[:, [0, self.size[1]-1, (self.size[0]-1)*self.size[1], self.size[0]*self.size[1]-1]]
        bounds = volume_bounds((transforms @ corners)[None, :, :3], self.model.margin_mm)
        sums = weights = None
        for start in range(0, len(frames), 16):
            stop = min(start + 16, len(frames))
            x = read_window(frames, start, stop, self.size, self.device)
            positions = (transforms[start:stop] @ points)[None, :, :3]
            part, weight = splat(positions, x[None], bounds, self.model.volume_shape, return_sums=True)
            sums = part if sums is None else sums + part
            weights = weight if weights is None else weights + weight
        field = self.model.deform(sums / weights.clamp_min(1e-6))
        prediction.field = field.cpu()
        prediction.bounds = tuple(x.cpu() for x in bounds)

    def predict_ddfs(self, frames, landmark, data_path_calib, output_dir=None):
        prediction = self.predict_scan(frames, data_path_calib)
        return export_ddfs(prediction, landmark, output_dir)


def export_ddfs(prediction, landmarks, output_dir=None):
    """Return GP,GL,LP,LL; optionally use .npy memmaps for large pixel arrays."""
    shape = (len(prediction.tool_global) - 1, 3, 480 * 640)
    if output_dir is None:
        gp, lp = np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gp, lp = [np.lib.format.open_memmap(output_dir / f'{key}.npy', mode='w+',
                                            dtype=np.float32, shape=shape) for key in ('GP', 'LP')]
    gl, ll = (x.cpu().numpy().astype(np.float32) for x in landmark_ddfs(prediction, landmarks))
    for i, start, global_ddf, local_ddf in iter_pixel_ddfs(prediction):
        stop = start + global_ddf.shape[-1]
        gp[i, :, start:stop] = global_ddf.cpu().numpy()
        lp[i, :, start:stop] = local_ddf.cpu().numpy()
    if output_dir is not None:
        gp.flush()
        lp.flush()
        np.save(output_dir / 'GL.npy', gl)
        np.save(output_dir / 'LL.npy', ll)
    return gp, gl, lp, ll
