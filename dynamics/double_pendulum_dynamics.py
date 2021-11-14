
import numpy as np

import torch
from torch.autograd.functional import jacobian, hessian

class DoublePendulum:
    """A class to represent a double pendulum and all of its system properties"""

    def __init__(self, m_matrix, l_vec):
        self.m = torch.tensor(m_matrix)
        self.lengths = torch.tensor(l_vec)
        self.g = torch.tensor(9.81)
        self.state_hist = None
