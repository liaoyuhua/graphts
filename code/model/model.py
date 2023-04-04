from torch import nn
import sampling
from layers import decoder


class AttrE2vec(nn.Module):
    def __init__(self):
        super().__init__()
