"""Per-scan RecON online optimization without access to scan pose labels."""
import copy
import torch
from torch.nn import functional as F

from geometry import image_points
from baselines.nr_rec_fus.models.volume import (normalize_points, splat,
                                                volume_bounds)
from .models import (chain_relative, correlation_loss, dof_to_matrix,
                     pair_inputs, scaled_motion, unscaled_motion)


def _motion_losses(model, inputs, target, config):
    prediction, _ = model.backbone(inputs)
    target_scaled = scaled_motion(target, config['angle_scale'])
    losses = {
        'distance': F.l1_loss(prediction[..., :3], target_scaled[..., :3]) * 3,
        'angle': F.l1_loss(prediction[..., 3:], target_scaled[..., 3:]) * 3,
        'correlation': correlation_loss(target_scaled, prediction),
    }
    return prediction, sum(losses.values()), losses


def _volume(frames, prediction, scale, config, selected=None, bounds=None):
    size = config['reconstruction_height'], config['reconstruction_width']
    small = F.interpolate(frames, size=size, mode='bilinear', align_corners=False)
    points = image_points(*size, device=frames.device, dtype=frames.dtype)
    transforms = chain_relative(dof_to_matrix(unscaled_motion(
        prediction, config['angle_scale'])))
    positions = (transforms @ (scale @ points))[:, :, :3]
    bounds = bounds or volume_bounds(positions, config['margin_mm'])
    if selected is not None:
        positions, small = positions[:, selected], small[:, selected]
    volume, occupancy = splat(positions, small, bounds, config['volume_shape'])
    return volume, occupancy, positions, bounds, small


def _content_loss(frames, prediction, scale, config):
    frame_count = frames.shape[1]
    every = max(int(round(1 / config['reconstruction_rate'])), 2)
    reconstruction_indices = torch.arange(0, frame_count, every, device=frames.device)
    held_indices = torch.tensor([i for i in range(frame_count)
                                 if i not in set(reconstruction_indices.tolist())],
                                device=frames.device)
    if not len(held_indices):
        return prediction.new_zeros(())
    volume, _, _, bounds, _ = _volume(frames, prediction, scale, config,
                                      selected=reconstruction_indices)
    size = config['reconstruction_height'], config['reconstruction_width']
    small = F.interpolate(frames, size=size, mode='bilinear', align_corners=False)
    points = image_points(*size, device=frames.device, dtype=frames.dtype)
    transforms = chain_relative(dof_to_matrix(unscaled_motion(
        prediction, config['angle_scale'])))
    positions = (transforms[:, held_indices] @ (scale @ points))[:, :, :3]
    grid = normalize_points(positions, bounds).reshape(len(frames), len(held_indices),
                                                       *size, 3)
    rendered = F.grid_sample(volume, grid, mode='bilinear', padding_mode='border',
                             align_corners=True).squeeze(1)
    return F.l1_loss(rendered, small[:, held_indices]) * config['fcc_weight']


class OnlineAdapter:
    def __init__(self, model, config, bank, device, reconstruction_scale,
                 initial_state=None):
        if not bank or bank.get('source_split') != 'train':
            raise ValueError('RecON online inference requires an embedded train-only PSC bank')
        self.model, self.config, self.bank, self.device = model, config, bank, device
        self.reconstruction_scale = reconstruction_scale
        self.initial = initial_state if initial_state is not None else copy.deepcopy(model.state_dict())

    def reset(self):
        self.model.load_state_dict(self.initial)

    def _bank_sample(self, current, generator):
        targets = self.bank['target_dof'].float()
        order = torch.randperm(len(targets), generator=generator).tolist()
        best_loss, best = float('inf'), order[0]
        for index in order[:self.config['psc_max_acquisition']]:
            length = min(current.shape[1], targets.shape[1])
            loss = float(correlation_loss(targets[index:index + 1, :length].to(current),
                                          current[:, :length]).detach().cpu())
            if loss < best_loss:
                best_loss, best = loss, index
            if loss <= 1 - self.config['psc_threshold']:
                break
        frames = self.bank['frames_uint8'][best:best + 1].to(self.device).float() / 255.
        target = targets[best:best + 1].to(self.device)
        return frames, target

    def predict(self, frames, scan_seed=0):
        self.reset()
        inputs = pair_inputs(frames)
        epochs = self.config['online_epochs']
        if epochs <= 0:
            self.model.eval()
            with torch.no_grad():
                return self.model.backbone(inputs)[0]
        psc_optimizer = torch.optim.Adam(self.model.backbone.parameters(),
                                         lr=self.config['online_psc_lr'])
        generator_optimizer = torch.optim.Adam(self.model.backbone.parameters(),
                                               lr=self.config['online_gas_fcc_lr'])
        discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(),
                                                   lr=self.config['online_gas_fcc_lr'])
        generator = torch.Generator().manual_seed(scan_seed)
        for epoch in range(1, epochs + 1):
            self.model.backbone.eval()
            with torch.no_grad():
                current = unscaled_motion(self.model.backbone(inputs)[0],
                                          self.config['angle_scale'])
            for _ in range(self.config['online_psc_steps']):
                bank_frames, bank_target = self._bank_sample(current, generator)
                bank_inputs = pair_inputs(bank_frames)
                self.model.backbone.train()
                psc_optimizer.zero_grad(set_to_none=True)
                _, loss, _ = _motion_losses(self.model, bank_inputs, bank_target, self.config)
                loss.backward()
                psc_optimizer.step()

            if epoch % self.config['discriminator_opt_cycle'] == 0:
                bank_frames, bank_target = self._bank_sample(current, generator)
                with torch.no_grad():
                    real_prediction = scaled_motion(bank_target, self.config['angle_scale'])
                    real, _, _, _, _ = _volume(bank_frames, real_prediction,
                                               self.reconstruction_scale, self.config)
                    fake_prediction = self.model.backbone(inputs)[0]
                    fake, _, _, _, _ = _volume(frames, fake_prediction,
                                               self.reconstruction_scale, self.config)
                self.model.discriminator.train()
                discriminator_optimizer.zero_grad(set_to_none=True)
                pred_real, pred_fake = self.model.discriminator(real), self.model.discriminator(fake)
                gas = pred_fake - pred_real
                norm = 2 * (real - fake).abs().mean().clamp_min(1e-6)
                discriminator_loss = (gas + gas.square() / norm).mean()
                discriminator_loss.backward()
                discriminator_optimizer.step()

            self.model.backbone.train()
            self.model.discriminator.eval()
            generator_optimizer.zero_grad(set_to_none=True)
            prediction = self.model.backbone(inputs)[0]
            fake, _, _, _, _ = _volume(frames, prediction, self.reconstruction_scale,
                                       self.config)
            gas_loss = -self.model.discriminator(fake).mean()
            fcc_loss = _content_loss(frames, prediction, self.reconstruction_scale, self.config)
            (gas_loss + fcc_loss).backward()
            generator_optimizer.step()
        self.model.eval()
        with torch.no_grad():
            return self.model.backbone(inputs)[0]
