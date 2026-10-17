"""Check count-prior correction and public candidate omission hypotheses."""
import unittest
import numpy as np
import torch
from evaluate_count_evidence import count_evidence


class CountEvidenceTests(unittest.TestCase):
    def test_posterior_equal_to_prior_adds_no_candidate_preference(self):
        prior=torch.arange(1,32,dtype=torch.float32);prior/=prior.sum()
        scores=count_evidence(prior.log()[None].repeat(2,1),prior.log(),torch.arange(2,33))
        torch.testing.assert_close(scores,torch.zeros(2,31),atol=1e-7,rtol=0)

    def test_omission_only_applies_to_routes_longer_than_four(self):
        probabilities=torch.arange(31,0,-1,dtype=torch.float32);probabilities/=probabilities.sum()
        prior=torch.full((31,),1/31)
        counts=torch.tensor([2,4,5,32])
        scores=count_evidence(probabilities.log()[None],prior.log(),counts,0.2835)
        ratios=probabilities/prior
        expected=[ratios[0],ratios[2],0.7165*ratios[3]+0.2835*ratios[2],0.7165*ratios[30]+0.2835*ratios[29]]
        torch.testing.assert_close(scores.exp()[0],torch.tensor(expected))
        with self.assertRaises(ValueError):count_evidence(probabilities.log()[None],prior.log(),torch.tensor([33]))


if __name__=='__main__':
    unittest.main()
