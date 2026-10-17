#!/usr/bin/env python3
"""Test point-only key-count evidence against a verified full spelling baseline."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prepare_waypoint_supervision import digest
from retrieval_model import build_encoder
from train_character import encode
from train_complete_geometry import metrics
from train_joint import load_checkpoint
from train_proxy import corpus
from waypoint_model import WaypointDecoder


def count_evidence(log_probabilities, log_prior, key_counts, omission=0.2835):
    """Integrate public candidate counts; correct the training count prior."""
    if (log_probabilities.ndim != 2 or log_probabilities.shape[1] != 31
            or log_prior.shape != (31,) or key_counts.ndim != 1 or torch.any((key_counts < 2) | (key_counts > 32))
            or not 0 <= omission < 1):
        raise ValueError('expected all counts from two to thirty-two and a valid omission prior')
    ratios = log_probabilities-log_prior
    ordinary = ratios[:, key_counts-2]
    if omission == 0:
        return ordinary
    deleted = ratios[:, (key_counts-3).clamp_min(0)]
    mixed = torch.logaddexp(ordinary+np.log1p(-omission), deleted+np.log(omission))
    return torch.where(key_counts[None]>4, mixed, ordinary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--counter',type=Path,required=True)
    parser.add_argument('--training-data',type=Path,required=True)
    parser.add_argument('--fusion-report',type=Path,required=True)
    parser.add_argument('--route-report',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('data/memory-proxy/validation.bin'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    saved,encoder_hash=load_checkpoint(args.checkpoint)
    counter,counter_hash=load_checkpoint(args.counter)
    fusion,route=[json.loads(path.read_text()) for path in [args.fusion_report,args.route_report]]
    rows,info=corpus(args.data)
    if (saved['metadata']['test_data_used'] or counter['metadata']['test_data_used']
            or counter['metadata']['encoder_sha256']!=encoder_hash or saved['words']!=counter['words']
            or len(saved['words'])!=60000 or saved['metadata']['validation_corpus']['sha256']!=info['sha256']):
        raise ValueError('count context or declared data differs')
    for report in [fusion,route]:
        if report['test_data_used'] or report['samples']!=len(rows) or report['full_vocabulary']!=60000 or report['development']['sha256']!=info['sha256']:
            raise ValueError('count selection requires every declared development row')
    if (not fusion['baseline_full_counts_verified'] or not fusion['inputs_unchanged_through_scoring']
            or fusion['route_report_sha256']!=digest(args.route_report)):
        raise ValueError('fusion baseline or candidate source differs')
    bound={str(path):digest(path) for path in [args.checkpoint,args.counter,args.training_data,args.fusion_report,args.route_report]}
    source_hashes={name:digest(Path(name)) for name in ['evaluate_count_evidence.py','waypoint_model.py','retrieval_model.py','large_encoder.py']}
    for path,report in [(args.fusion_report,fusion),(args.route_report,route)]:
        if digest(path.with_suffix('.scores.npz'))!=report['score_cache_sha256']:
            raise ValueError('source scores differ')
    with np.load(args.fusion_report.with_suffix('.scores.npz'),allow_pickle=False) as data:
        costs,candidates,selected=[data[name] for name in ['spelling_costs','candidates','selected_rows']]
    with np.load(args.route_report.with_suffix('.scores.npz'),allow_pickle=False) as data:
        if not np.array_equal(candidates,data['candidates']) or not np.array_equal(selected,data['selected_rows']):
            raise ValueError('candidate lists or row identities differ')
        similarity=data['similarities']
    if candidates.shape!=(12000,128) or not np.array_equal(selected,np.arange(12000)):
        raise ValueError('expected the complete twelve-thousand-row natural shortlist')
    training=counter['metadata']['data']
    if digest(args.training_data)!=training['data_sha256']:
        raise ValueError('count-prior training data differs')
    with np.load(args.training_data,allow_pickle=False) as data:
        # The internal holdout supplies no marginal normalization statistics.
        frequencies=np.bincount(data['counts'][training['internal_holdout']:],minlength=33)[2:33].astype(np.float64)
    if np.any(frequencies==0):raise ValueError('every count must have training examples')
    log_prior=torch.tensor(np.log(frequencies/frequencies.sum()),dtype=torch.float32)
    encoder=build_encoder(saved['metadata']['arguments']['architecture']).to('mps').eval().requires_grad_(False)
    encoder.load_state_dict(saved['model'])
    config=counter['metadata']['arguments']
    model=WaypointDecoder(getattr(encoder,'input_channels',8),config['width'],config['layers']).to('mps').eval().requires_grad_(False)
    model.load_state_dict(counter['model'])
    probabilities=[]
    with torch.no_grad():
        for start in range(0,len(rows),192):
            points=torch.tensor(np.array(rows['points'][start:start+192]),device='mps')
            probabilities.append(model.count_logits(model.encode_memory(*encode(encoder,points))).log_softmax(1).cpu())
    probabilities=torch.cat(probabilities)
    key_counts=torch.tensor([sum(i==0 or c!=word[i-1] for i,c in enumerate(word)) for word in saved['words']])
    # Every word shares one of the thirty-one public count hypotheses.
    # Evaluate those once, then gather the unchanged natural candidate scores.
    evidence_by_count=count_evidence(probabilities,log_prior,torch.arange(2,33))
    ids=torch.tensor(candidates,dtype=torch.long)
    evidence=evidence_by_count.gather(1,key_counts[ids]-2)
    fixed=fusion['fixed_spelling_settings']
    sizes=torch.tensor([len(word)+1 for word in saved['words']])[ids].float()
    baseline=-torch.tensor(costs)/sizes.pow(fixed['length_power'])+32*fixed['proxy_weight']*torch.tensor(similarity)
    labels=torch.tensor(np.array(rows['label'],dtype=np.int64));difficulty=torch.tensor(np.array(rows['difficulty'],dtype=np.int64))
    trials=[{'count_weight':weight,'validation':metrics(baseline+weight*evidence,ids,labels,difficulty)}
            for weight in [0.0,0.1,0.25,0.5,1.0,2.0,4.0]]
    for name,values in trials[0]['validation'].items():
        for key in ['correct_top1','correct_top5']:
            if values[key]!=fusion['trials'][0]['validation'][name][key]:raise ValueError('zero-count baseline differs')
    if any(digest(Path(path))!=value for path,value in bound.items()) or any(digest(Path(name))!=value for name,value in source_hashes.items()):
        raise ValueError('count model, source, or evidence changed while scoring')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output.with_suffix('.scores.npz'),count_log_probabilities=probabilities.numpy(),training_log_prior=log_prior.numpy())
    report={'test_data_used':False,'development':info,'samples':len(rows),'full_vocabulary':60000,
            'encoder_sha256':encoder_hash,'counter_sha256':counter_hash,'counter_epoch':counter['epoch'],
            'inputs_sha256':bound,'source_sha256':source_hashes,'baseline_counts_verified':True,
            'prior_source':'Training rows only; internal holdout excluded','training_count_prior':log_prior.exp().tolist(),
            'omission_prior':0.2835,'selected':max(trials,key=lambda row:row['validation']['memory']['correct_top1']),'trials':trials,
            'native_supported':False,'limits':'Full development with identical natural128 candidates. Query inputs are points only; all31count classes remain available. Counts from candidate spellings are public hypotheses. No true query count, target, writer, or difficulty enters prediction; no benchmark or confirmation evaluation.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['selected']),flush=True)


if __name__=='__main__':
    main()
