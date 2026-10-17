#!/usr/bin/env python3
"""Learn unrounded corner geometry using fresh training writers only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from prepare_complete_geometry import validate_roles
from prepare_waypoint_supervision import digest
from retrieval_model import build_encoder
from train_character import encode
from train_joint import load_checkpoint
from waypoint_model import WaypointDecoder, supervised_loss


@torch.no_grad()
def evaluate(model, encoder, points, targets, counts, size, batch):
    model.eval()
    total, correct, squared, coordinates = 0.0, 0, 0.0, 0
    for start in range(0,size,batch):
        end = min(start+batch,size)
        path, corner, count = [torch.tensor(array[start:end],device='mps') for array in [points,targets,counts]]
        memory = model.encode_memory(*encode(encoder,path))
        loss, _, _ = supervised_loss(model,memory,path,corner,count)
        means, _, valid = model.conditioned(memory,path,count)
        total += float(loss)*(end-start)
        correct += int((model.count_logits(memory).argmax(1)+2 == count).sum())
        squared += float(((means-corner).square()*valid[...,None]).sum())
        coordinates += int(valid.sum())*2
    return {'conditional_loss':total/size,'count_accuracy':correct/size,
            'conditional_waypoint_mse':squared/coordinates,'samples':size,
            'recognition_accuracy_measured':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--width',type=int,choices=[192,384],default=192)
    parser.add_argument('--layers',type=int,choices=[3,6],default=3)
    parser.add_argument('--seed',type=int,default=900941)
    args = parser.parse_args()
    if args.output.exists() or min(args.epochs,args.steps,args.batch)<1:
        raise ValueError('use a new output directory and positive training counts')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saved, context_hash = load_checkpoint(args.checkpoint)
    data_info = json.loads(args.data.with_suffix('.json').read_text())
    if (saved['metadata']['test_data_used'] or data_info['test_data_used']
            or data_info['split']!='train' or data_info['checkpoint_sha256']!=context_hash
            or digest(args.data)!=data_info['data_sha256'] or len(saved['words'])!=60000
            or data_info['validation_corpus']['sha256']!=saved['metadata']['validation_corpus']['sha256']):
        raise ValueError('waypoint training data, context, or declared development differs')
    validate_roles(data_info['corpora'],data_info['validation_corpus'],'train',data_info['samples'])
    with np.load(args.data,allow_pickle=False) as data:
        points,targets,counts=[data[name] for name in ['points','waypoints','counts']]
    size=data_info['samples'];heldout=data_info['internal_holdout']
    if (points.shape!=(size,64,2) or targets.shape!=(size,32,2) or counts.shape!=(size,)
            or not np.isfinite(points).all() or not np.isfinite(targets).all()
            or np.any((counts<2)|(counts>32)) or not 0<heldout<size):
        raise ValueError('invalid waypoint arrays or internal split')
    padding=np.arange(32)[None]>=counts[:,None]
    if np.any(targets[padding]!=0):
        raise ValueError('unused waypoint targets must remain padded with zero')
    encoder=build_encoder(saved['metadata']['arguments']['architecture']).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(saved['model'])
    model=WaypointDecoder(getattr(encoder,'input_channels',8),args.width,args.layers).to('mps')
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=0.00001)
    metadata={'test_data_used':False,'encoder_sha256':context_hash,'data':data_info,
              'parameters':sum(p.numel() for p in model.parameters()),
              'selection':'Internal training-writer conditional loss only; no development word-recognition score',
              'query_inputs':'Normalized stroke points; inference enumerates every count from2to32. True counts and corner locations supervise training only.',
              'source_sha256':{name:digest(Path(name)) for name in ['train_waypoints.py','waypoint_model.py','retrieval_model.py','large_encoder.py']},
              'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    args.output.mkdir(parents=True)
    (args.output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    best=float('inf');began=time.monotonic()
    for epoch in range(args.epochs+1):
        total=0.0
        if epoch:
            model.train()
            for _ in range(args.steps):
                ids=rng.integers(heldout,size,size=args.batch)
                path,corner,count=[torch.tensor(array[ids],device='mps') for array in [points,targets,counts]]
                memory=model.encode_memory(*encode(encoder,path))
                loss,_,_=supervised_loss(model,memory,path,corner,count)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite corner-training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step();total+=float(loss.detach())
            scheduler.step()
        validation=evaluate(model,encoder,points,targets,counts,heldout,args.batch)
        report={'epoch':epoch,'training_loss':total/max(args.steps,1),'internal_holdout':validation,
                'seconds':time.monotonic()-began}
        with (args.output/'history.jsonl').open('a') as output:output.write(json.dumps(report)+'\n')
        checkpoint={'model':{name:value.detach().cpu() for name,value in model.state_dict().items()},
                    'epoch':epoch,'metadata':metadata,'words':saved['words']}
        temporary=args.output/'latest.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(args.output/'latest.pt')
        if validation['conditional_loss']<best:
            best=validation['conditional_loss']
            temporary=args.output/'best.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(args.output/'best.pt')
            (args.output/'best-metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
