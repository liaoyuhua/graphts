import torch
from ..aggregators import BaseAggregator


class EdgeSampleAggregate(torch.nn.Module):
    
    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        aggregator: BaseAggregator,
        encoder: torch.nn.Module,
    ):
        """Inits EdgeSampleAggregate."""
        super().__init__()

        self._edge_dim = edge_dim
        self._node_dim = node_dim

        self._agg = aggregator
        self._enc = encoder

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._agg.to(self._device)
        self._enc.to(self._device)