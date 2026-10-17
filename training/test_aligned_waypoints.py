"""Check ordered corner hypotheses and supervision-only alignment targets."""
import unittest
import torch

from aligned_waypoint_model import AlignedWaypointDecoder,aligned_loss
from waypoint_model import WaypointDecoder


class AlignedWaypointTests(unittest.TestCase):
    def test_warm_start_retains_count_logits_and_initial_locations(self):
        torch.manual_seed(125)
        old=WaypointDecoder(width=48,layers=1).eval()
        new=AlignedWaypointDecoder(width=48,layers=1).eval();new.warm_start(old.state_dict())
        sequence,raw,embedding=torch.randn(2,128,32),torch.randn(2,8,64),torch.randn(2,128)
        points=torch.randn(2,64,2);counts=torch.tensor([5,9])
        with torch.no_grad():
            first=old.encode_memory(sequence,raw,embedding);second=new.encode_memory(sequence,raw,embedding)
            torch.testing.assert_close(new.count_logits(second),old.count_logits(first),atol=0,rtol=0)
            means,_,valid,positions=new.aligned(second,points,counts)
            torch.testing.assert_close(means,old.conditioned(first,points,counts)[0],atol=1e-5,rtol=1e-6)
            self.assertTrue(torch.all(positions[:,1:]>=positions[:,:-1]))
            self.assertTrue(torch.all(positions[:,0]==0))
            self.assertTrue(torch.all(positions.gather(1,(counts-1)[:,None])==1))
            result=new.infer(second,points,chunk=7)
            self.assertEqual(result['means'].shape,(2,31,32,2))
            self.assertEqual(result['valid'].sum(1).tolist(),list(range(2,33)))

    def test_alignment_and_location_heads_learn_from_nonuniform_corners(self):
        model=AlignedWaypointDecoder(width=48,layers=1)
        memory,points=torch.randn(2,64,48),torch.randn(2,64,2)
        counts=torch.tensor([4,4]);targets=torch.randn(2,32,2)
        arc=torch.zeros(2,32);arc[:,:4]=torch.tensor([0.,.1,.8,1.])
        loss,_,_=aligned_loss(model,memory,points,targets,counts,arc)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.gaps.weight.grad.norm()),0)
        self.assertGreater(float(model.location.weight.grad.norm()),0)
        with self.assertRaises(ValueError):model.warm_start({'bad':torch.zeros(1)})


if __name__=='__main__':
    unittest.main()
