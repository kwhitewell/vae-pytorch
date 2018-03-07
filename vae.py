import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from typing import Tuple

class Encoder(nn.Module):
  def __init__(
    self,
    n_in: int,
    n_f: int,
    n_z: int,
  ) -> None:
    super(Encoder, self).__init__()

    self.l1 = nn.Linear(n_in, n_f)
    self.l2_mu = nn.Linear(n_f, n_z)
    self.l2_var = nn.Linear(n_f, n_z)
    
    self.n_in = n_in
    self.n_f = n_f
    self.n_z = n_z

  def forward(
    self,
    x: Variable,
  ) -> Tuple[Variable, Variable]:

    h = F.relu(self.l1(x))

    mu = self.l2_mu(h)
    var = self.l2_var(h)

    return mu, var


class Decoder(nn.Module):
  def __init__(
    self,
    n_z: int,
    n_f: int,
    n_out: int,
  ) -> None:
    super(Decoder, self).__init__()

    self.l1 = nn.Linear(n_z, n_f)
    self.l2 = nn.Linear(n_f, n_out)

    self.n_z = n_z
    self.n_f = n_f
    self.n_out = n_out

  def forward(
    self,
    z: Variable,
  ) -> Variable:

    return self.l2(F.relu(self.l1(z)))
  
