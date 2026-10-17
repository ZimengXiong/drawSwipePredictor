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
from aligned_waypoint_model import AlignedWaypointDecoder, aligned_loss


@torch.no_grad()
def evaluate(model, encoder, points, targets, counts, size, batch, progress=None):
    model.eval()
    total, correct, squared, coordinates = 0.0, 0, 0.0, 0
    query_mse, arc_mse = 0.0, 0.0
    for start in range(0,size,batch):
        end = min(start+batch,size)
        path, corner, count = [torch.tensor(array[start:end],device='mps') for array in [points,targets,counts]]
        memory = model.encode_memory(*encode(encoder,path))
        if progress is None:
            loss, _, _ = supervised_loss(model,memory,path,corner,count)
            means, _, valid = model.conditioned(memory,path,count)
        else:
            arc=torch.tensor(progress[start:end],device='mps')
            loss,_,_=aligned_loss(model,memory,path,corner,count,arc)
            means,_,valid,positions=model.aligned(memory,path,count)
            arc_mse += float((((positions-arc).square()*valid).sum(1)/count).sum())
        query_mse += float((((means-corner).square()*valid[...,None]).sum((1,2))/(2*count)).sum())
        total += float(loss)*(end-start)
        correct += int((model.count_logits(memory).argmax(1)+2 == count).sum())
        squared += float(((means-corner).square()*valid[...,None]).sum())
        coordinates += int(valid.sum())*2
    return {'conditional_loss':total/size,'count_accuracy':correct/size,
            'conditional_waypoint_mse':squared/coordinates,'samples':size,
            'mean_query_waypoint_mse':query_mse/size,'arc_position_mse':arc_mse/size if progress is not None else None,
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
    parser.add_argument('--with-progress',action='store_true')
    parser.add_argument('--init-model',type=Path)
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
        progress=data['arc_positions'] if args.with_progress else None
    size=data_info['samples'];heldout=data_info['internal_holdout']
    if (points.shape!=(size,64,2) or targets.shape!=(size,32,2) or counts.shape!=(size,)
            or not np.isfinite(points).all() or not np.isfinite(targets).all()
            or np.any((counts<2)|(counts>32)) or not 0<heldout<size):
        raise ValueError('invalid waypoint arrays or internal split')
    padding=np.arange(32)[None]>=counts[:,None]
    if np.any(targets[padding]!=0):
        raise ValueError('unused waypoint targets must remain padded with zero')
    if progress is not None:
        if (data_info.get('target_schema')!='waypoints_arc_v2' or progress.shape!=(size,32)
                or not np.isfinite(progress).all() or np.any(progress[padding]!=0)
                or np.any(progress[:,0]!=0) or np.any(progress[np.arange(size),counts-1]!=1)):
            raise ValueError('invalid ordered alignment supervision')
        active_pairs=np.arange(31)[None]<counts[:,None]-1
        if np.any((np.diff(progress,axis=1)<0)&active_pairs):
            raise ValueError('arc targets must stay in chronological order')
    if args.init_model and not args.with_progress:
        raise ValueError('warm starts require the versioned alignment model')
    encoder=build_encoder(saved['metadata']['arguments']['architecture']).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(saved['model'])
    factory=AlignedWaypointDecoder if args.with_progress else WaypointDecoder
    model=factory(getattr(encoder,'input_channels',8),args.width,args.layers).to('mps')
    initial_hash=None
    if args.init_model:
        initial,initial_hash=load_checkpoint(args.init_model)
        if (initial['metadata']['test_data_used'] or initial['metadata']['encoder_sha256']!=context_hash
                or initial['words']!=saved['words']):
            raise ValueError('corner initialization belongs to a different context')
        previous=initial['metadata']['data']['data_sha256']
        if previous!=data_info['data_sha256'] and (previous!=data_info.get('reference_data_sha256') or not data_info.get('base_arrays_identical_to_reference')):
            raise ValueError('warm start requires unchanged query and corner arrays')
        model.warm_start(initial['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=0.00001)
    metadata={'test_data_used':False,'encoder_sha256':context_hash,'data':data_info,
              'parameters':sum(p.numel() for p in model.parameters()),
              'model_type':'aligned_waypoints_v2' if args.with_progress else 'waypoints_v1',
              'initialization_sha256':initial_hash,
              'selection':'Internal mean-query location MSE plus ten times arc MSE' if args.with_progress else 'Internal training-writer conditional loss only; no development word-recognition score',
              'query_inputs':'Normalized stroke points; inference enumerates every count from2to32. True counts and corner locations supervise training only.',
              'source_sha256':{name:digest(Path(name)) for name in ['train_waypoints.py','waypoint_model.py','aligned_waypoint_model.py','retrieval_model.py','large_encoder.py']},
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
                if progress is None:
                    loss,_,_=supervised_loss(model,memory,path,corner,count)
                else:
                    arc=torch.tensor(progress[ids],device='mps')
                    loss,_,_=aligned_loss(model,memory,path,corner,count,arc)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite corner-training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step();total+=float(loss.detach())
            scheduler.step()
        validation=evaluate(model,encoder,points,targets,counts,heldout,args.batch,progress)
        report={'epoch':epoch,'training_loss':total/max(args.steps,1),'internal_holdout':validation,
                'seconds':time.monotonic()-began}
        with (args.output/'history.jsonl').open('a') as output:output.write(json.dumps(report)+'\n')
        checkpoint={'model':{name:value.detach().cpu() for name,value in model.state_dict().items()},
                    'epoch':epoch,'metadata':metadata,'words':saved['words']}
        temporary=args.output/'latest.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(args.output/'latest.pt')
        selection=(validation['mean_query_waypoint_mse']+10*validation['arc_position_mse']) if args.with_progress else validation['conditional_loss']
        if selection<best:
            best=selection
            temporary=args.output/'best.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(args.output/'best.pt')
            (args.output/'best-metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
