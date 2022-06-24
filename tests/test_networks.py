import sys
import unittest
import torch
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.networks import AttentionNetwork, DeepFM, FACNN


# class TestATTCNN(unittest.TestCase):

#     def setUp(self):
#         # torch.cuda.set_device(0)
#         self.model = AttentionNetwork()
#         self.model.eval()
#         # self.model.cuda(0)
#         # torch.manual_seed(0)
#         # # instead of zero init for score tensors use random init
#         # self.model.score_fr[6].weight.data.random_()
#         # self.model.score_fr[6].bias.data.random_()
#         # self.model.score_pool3.weight.data.random_()
#         # self.model.score_pool3.bias.data.random_()
#         # self.model.score_pool4.weight.data.random_()
#         # self.model.score_pool4.bias.data.random_()
#         self.x = torch.rand((1, 1, 8760, 32))
        

#     def testForward(self):
#         print("testForward")
#         # self.assertEqual(torch.cuda.is_available(), True)
#         result_foward = self.model.forward(self.x)
#         # torch.Size([1, 1, 8729, 1])
#         self.assertEqual(result_foward.shape.numel(), 8760)

class TestDFM(unittest.TestCase):
    def setUp(self):
        # torch.cuda.set_device(0)
        date_range = 100
        category = 33
        # ()
        # self.model = DeepFM((32,) ,embed_dim=16, mlp_dims=(8, 8), dropout=0.2)
        # # embedding 35, 16
        
        # self.model.cuda(0)
        # torch.manual_seed(0)
        # # instead of zero init for score tensors use random init
        # self.model.score_fr[6].weight.data.random_()
        # self.model.score_fr[6].bias.data.random_()
        # self.model.score_pool3.weight.data.random_()
        # self.model.score_pool3.bias.data.random_()
        # self.model.score_pool4.weight.data.random_()
        # self.model.score_pool4.bias.data.random_()
        self.x = torch.rand((32, 16))
        # self.embed_info = torch.rand((32, 16))
        # self.embed_info = torch.tensor([i for i in range(32)])

        self.model = FACNN()
        self.model.eval()
        

    def testForward(self):
        print("testForward")
        # self.assertEqual(torch.cuda.is_available(), True)
        # result_foward = self.model(self.embed_info)
        batch_num = 10
        data =  torch.tensor([[1 for i in range(1, 33)]for _ in range(batch_num)])
        # result_foward = self.model(self.embed_info)
        result = self.model(data)



if __name__ == "__main__":
    unittest.main()