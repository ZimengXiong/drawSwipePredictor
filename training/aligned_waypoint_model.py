"""Recover corners through supervised monotone alignment to the observed stroke."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from waypoint_model import WaypointDecoder


def sample_arc(points, positions):
    location=positions.clamp(0,1)*63
    lo=location.long().clamp_max(62)
    first=points.gather(1,lo[...,None].expand(-1,-1,2))
    last=points.gather(1,(lo+1)[...,None].expand(-1,-1,2))
    return first+(location-lo)[...,None]*(last-first)


class AlignedWaypointDecoder(WaypointDecoder):
    def __init__(self,raw_channels=8,width=192,layers=3):
        super().__init__(raw_channels,width,layers)
        self.gaps=nn.Linear(width,1)
        self.arc_encoding=nn.Linear(9,width,bias=False)
        nn.init.zeros_(self.gaps.weight);nn.init.zeros_(self.gaps.bias)
        nn.init.zeros_(self.arc_encoding.weight)

    def warm_start(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        expected={'gaps.weight','gaps.bias','arc_encoding.weight'}
        if unexpected or set(missing) not in [set(),expected]:
            raise ValueError('incompatible corner-model initialization')

    def encode_memory(self,sequence,raw,embedding):
        memory=super().encode_memory(sequence,raw,embedding)
        progress=torch.linspace(0,1,64,device=memory.device)
        features=[progress]
        for frequency in [1,2,4,8]:
            features.extend([(math.pi*frequency*progress).sin(),(math.pi*frequency*progress).cos()])
        return memory+self.arc_encoding(torch.stack(features,dim=1))[None]

    def aligned(self,memory,points,hypotheses):
        if hypotheses.shape!=(len(memory),) or torch.any((hypotheses<2)|(hypotheses>32)):
            raise ValueError('each count hypothesis must be between two and thirty-two')
        slots=torch.arange(32,device=memory.device)
        valid=slots[None]<hypotheses[:,None]
        fraction=(slots[None]/(hypotheses[:,None]-1)).clamp_max(1)
        base=sample_arc(points,fraction)
        query=(self.slots(slots)[None]+self.counts(hypotheses)[:,None]
               +self.base_projection(torch.cat([base,fraction[...,None]],dim=2)))
        hidden=self.decoder(query,memory,tgt_key_padding_mask=~valid)
        valid_gaps=slots[None]<hypotheses[:,None]-1
        weights=self.gaps(hidden).squeeze(-1).masked_fill(~valid_gaps,-torch.inf).softmax(1)
        positions=F.pad(weights.cumsum(1)[:,:31],(1,0)).clamp(0,1)
        positions=torch.where(slots[None]>=hypotheses[:,None]-1,1.0,positions)
        means=sample_arc(points,positions)+self.location(hidden)
        scales=self.log_scale(hidden).clamp(math.log(0.03),1)
        return means,scales,valid,positions

    def conditioned(self,memory,points,hypotheses):
        means,scales,valid,_=self.aligned(memory,points,hypotheses)
        return means,scales,valid


def aligned_loss(model,memory,points,targets,counts,arc_targets):
    means,log_scale,valid,positions=model.aligned(memory,points,counts)
    gaussian=0.5*((targets-means)*(-log_scale).exp()).square()+log_scale
    density=((gaussian*valid[...,None]).sum((1,2))/(2*counts)).mean()
    location=(((means-targets).square()*valid[...,None]).sum((1,2))/(2*counts)).mean()
    alignment=(((positions-arc_targets).square()*valid).sum(1)/counts).mean()
    classification=F.cross_entropy(model.count_logits(memory),counts-2)
    return density+2*location+25*alignment+0.5*classification,location,alignment
