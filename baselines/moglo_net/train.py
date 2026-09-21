"""MoGLo-Net training on TUS-REC2024 using the public loss formulation."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from geometry import calibration_tensors, image_points
from training import seed_all, select_device, run_training
from baselines.nr_rec_fus.models.rigid_pose import se3_from_vec
from .dataset import MoGLoDataset
from .models import MoGLoNet, transforms_to_motion


def motion_labels(tforms, spatial):
    relative_tool = torch.linalg.inv(tforms[:, :-1]) @ tforms[:, 1:]
    relative_image = torch.linalg.inv(spatial) @ relative_tool @ spatial
    return transforms_to_motion(relative_image)


def objective(model, batch, scale, spatial, config, device):
    frames = batch['frames'].to(device).unsqueeze(2)
    tforms = batch['tforms'].to(device)
    scale_motion = torch.as_tensor(config['motion_scale'], device=device, dtype=frames.dtype)
    target = motion_labels(tforms, spatial) / scale_motion
    predictions, embedding, _ = model(frames)
    mme = (F.smooth_l1_loss(predictions[:, 0], target) +
           F.smooth_l1_loss(predictions[:, 1], target)) / 2
    correlation = (1 - F.cosine_similarity(predictions[:, 0].flatten(1), target.flatten(1), dim=1).mean() +
                   1 - F.cosine_similarity(predictions[:, 1].flatten(1), target.flatten(1), dim=1).mean()) / 2
    # Public implementation samples triplets from the final pair embedding.
    triplet = predictions.new_zeros(())
    if len(frames) >= 3:
        emb = embedding[:, -1]
        y = target[:, -1]
        terms = []
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                for k in range(j + 1, len(frames)):
                    d_y_ij = 1 - F.cosine_similarity(y[i:i + 1], y[j:j + 1])
                    d_y_ik = 1 - F.cosine_similarity(y[i:i + 1], y[k:k + 1])
                    sign = torch.sign(d_y_ik - d_y_ij)
                    d_v_ij = torch.linalg.vector_norm(emb[i] - emb[j])
                    d_v_ik = torch.linalg.vector_norm(emb[i] - emb[k])
                    terms.append(F.relu(sign * (d_v_ij - d_v_ik) + config['triplet_margin']))
        if terms:
            triplet = torch.stack(terms).mean()
    loss = (config['mme_weight'] * mme + config['correlation_weight'] * correlation +
            config['triplet_weight'] * triplet)
    with torch.no_grad():
        predicted_motion = predictions[:, 1] * scale_motion
        true_motion = target * scale_motion
        points = image_points(*frames.shape[-2:], device=device, dtype=frames.dtype)
        predicted_t = se3_from_vec(predicted_motion)
        true_t = se3_from_vec(true_motion)
        predicted_points = (predicted_t @ (scale @ points))[:, :, :3]
        true_points = (true_t @ (scale @ points))[:, :, :3]
        distance = torch.linalg.vector_norm(predicted_points - true_points, dim=2).mean()
    return loss, {'distance_mm': distance, 'MME': mme, 'COR': correlation,
                  'TRI': triplet, 'motion_mae': (predicted_motion - true_motion).abs().mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='baselines/moglo_net/configs/tus_rec2024.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default=None)
    p.add_argument('--resume', default='')
    p.add_argument('--epochs', type=int)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--steps', '--max-steps', dest='max_steps', type=int, default=0)
    p.add_argument('--val-steps', type=int, default=0)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    if a.epochs is not None:
        config['epochs'] = a.epochs
    config['method'] = 'moglo_net'
    seed_all(config['seed'])
    device = select_device(a.device)
    scale, spatial = calibration_tensors(Path(config['root']) / 'calib_matrix.csv', device,
                                         size=(config['height'], config['width']))
    train = MoGLoDataset(config['root'], 'train', config['preprocessed'], config['num_samples'],
                         config['sample_range'], (config['height'], config['width']), config['seed'], config['val_windows'])
    val = MoGLoDataset(config['root'], 'val', config['preprocessed'], config['num_samples'],
                       config['sample_range'], (config['height'], config['width']), config['seed'], config['val_windows'])
    model = MoGLoNet(config['num_samples']).to(device)
    fn = lambda m, b: objective(m, b, scale, spatial, config, device)
    run_training(model, train, val, fn, config, device, a.output or config['output'], config['epochs'],
                 config['batch_size'], config['lr'], a.workers, a.resume, a.max_steps, a.val_steps,
                 config['weight_decay'])


if __name__ == '__main__':
    main()
