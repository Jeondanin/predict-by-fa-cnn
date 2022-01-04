import sys
import unittest
import torch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.networks import AttentionNetwork


class TestATTCNN(unittest.TestCase):

    def setUp(self):
        # torch.cuda.set_device(0)
        self.model = AttentionNetwork()
        self.model.eval()
        # self.model.cuda(0)
        # torch.manual_seed(0)
        # # instead of zero init for score tensors use random init
        # self.model.score_fr[6].weight.data.random_()
        # self.model.score_fr[6].bias.data.random_()
        # self.model.score_pool3.weight.data.random_()
        # self.model.score_pool3.bias.data.random_()
        # self.model.score_pool4.weight.data.random_()
        # self.model.score_pool4.bias.data.random_()
        self.x = torch.rand((1, 1, 8760, 32))
        

    def testForward(self):
        print("testForward")
        # self.assertEqual(torch.cuda.is_available(), True)
        result_foward = self.model.forward(self.x)
        # torch.Size([1, 1, 8729, 1])
        self.assertEqual(result_foward.shape.numel(), 8760)



if __name__ == "__main__":
    unittest.main()