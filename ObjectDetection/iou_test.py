import sys 
import unittest 
import torch 
from iou import intersection_over_union 

class Test(unittest.TestCase):
    def setUp(self):
        # test case we want to run
        self.t1_box1 = torch.tensor([0.8, 0.1, 0.2, 0.2])
        self.t1_box2 = torch.tensor([0.9, 0.2, 0.2, 0.2])
        self.t1_correct_iou = 1/7
        self.epsilon = 0.001

    def test_both_inside_cell_shares_area(self):
        iou = intersection_over_union(self.t1_box1, 
        self.t1_box2, box_format="midpoint")

        self.assertTrue((torch.abs(iou - self.t1_correct_iou) < self.epsilon))


if __name__ == "__main__":
    print("Running IOUs")
    unittest.main()