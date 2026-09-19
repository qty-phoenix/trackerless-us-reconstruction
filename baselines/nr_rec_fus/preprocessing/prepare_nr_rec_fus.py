"""Validate and summarize TUS-REC2024 inputs; data are referenced in place."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',default='data/tus-rec2024'); p.add_argument('--out',default='baselines/nr_rec_fus/results'); a=p.parse_args(); root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 rows={s:[json.loads(x) for x in (root/'preprocessed/longterm'/f'{s}.jsonl').read_text().splitlines() if x] for s in ('train','val')}
 meta={'method':'NR-Rec-FUS','dataset':'TUS-REC2024','train_scans':len(rows['train']),'val_scans':len(rows['val']),'data_in_place':True,'resolution':[120,160]}; (out/'dataset_summary.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
