import unittest
import torch
from train_door_dp import require_finite_loss


class FiniteLossTests(unittest.TestCase):
    def test_nan_and_infinity_stop_before_backward(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            loss=torch.tensor(value,requires_grad=True)
            with self.assertRaisesRegex(FloatingPointError,'1001'):
                require_finite_loss(loss,1001)
            self.assertIsNone(loss.grad)

    def test_finite_loss_preserves_gradient_and_rng(self):
        x=torch.tensor([.3,.7],requires_grad=True);loss=x.square().mean()
        rng=torch.get_rng_state().clone()
        require_finite_loss(loss,1001);loss.backward()
        self.assertTrue(torch.equal(x.grad,torch.tensor([.3,.7])))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
