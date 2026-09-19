import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .datasets.nr_rec_dataset import NRRecDataset
from .models.nr_rec_fus import NRRecFUS
def relative(t): return torch.linalg.inv(t[:,0:1])@t[:,1:]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='baselines/nr_rec_fus/configs/tus_rec2024.json'); p.add_argument('--steps',type=int,default=0); p.add_argument('--device',default='cuda'); a=p.parse_args(); c=json.loads(Path(a.config).read_text()); dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); root=c['root']; ds=NRRecDataset(root,'train',c['preprocessed'],c['num_samples'],c['sample_range'],(c['height'],c['width'])); dl=DataLoader(ds,batch_size=c['batch_size'],shuffle=True,num_workers=0); model=NRRecFUS(c['num_samples']).to(dev); opt=torch.optim.Adam(model.parameters(),lr=c['lr']); model.train(); steps=0
 for ep in range(c['epochs']):
  for b in dl:
   x=b['frames'].to(dev); out=model(x); gt=relative(b['tforms'].to(dev)); pose_loss=torch.nn.functional.smooth_l1_loss(out['pose'],gt)
   flow=out['flow']; dx=(flow[..., :,1:]-flow[..., :, :-1]).abs().mean(); dy=(flow[..., 1:,:]-flow[..., :-1,:]).abs().mean(); smooth=dx+dy; photo=(out['warped']-x[:,0:1]).abs().mean(); loss=pose_loss+photo+c['smooth_weight']*smooth; opt.zero_grad(); loss.backward(); opt.step(); steps+=1
   if steps%10==0: print(f'epoch={ep} step={steps} loss={loss.item():.5f} pose={pose_loss.item():.5f}',flush=True)
   if a.steps and steps>=a.steps: return
if __name__=='__main__': main()
