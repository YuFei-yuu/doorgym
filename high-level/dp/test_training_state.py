"""Resume reproduces actual stochastic updates, including crossing epoch boundaries."""
import copy
import random
import unittest

import numpy as np
import torch

from training_state import ResumableBatchSampler, capture_rng, restore_rng


class ResumeTest(unittest.TestCase):
    def test_stochastic_updates(self):
        torch.set_num_threads(2)
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sampler = ResumableBatchSampler(23, 4, 43)
        loader = torch.utils.data.DataLoader(torch.arange(69).reshape(23,3).float(),
                    batch_sampler=sampler, generator=torch.Generator().manual_seed(0))

        def advance(count):
            iterator = iter(loader)
            for _ in range(count):
                try: x = next(iterator)
                except StopIteration: iterator = iter(loader); x = next(iterator)
                y = torch.randn(len(x),2) + np.random.rand() + random.random()
                loss = (model(x)-y).square().mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        advance(3)
        saved = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   sampler=sampler.state_dict(), rng=capture_rng()))
        advance(9)
        expected = copy.deepcopy(model.state_dict())
        # Simulate process initialization consuming RNG before restoration.
        torch.randn(100); np.random.rand(50); random.random()
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler']); restore_rng(saved['rng'])
        advance(9)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, expected[k]), k)

    def test_reject_changed_data(self):
        sampler = ResumableBatchSampler(23, 4)
        next(iter(sampler))
        with self.assertRaises(ValueError):
            ResumableBatchSampler(24,4).load_state_dict(sampler.state_dict())


if __name__ == '__main__': unittest.main()
