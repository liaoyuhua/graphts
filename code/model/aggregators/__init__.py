from argparse import Namespace
from avg import *
from base import *
from rnn import *


def build_aggregator(name: str, hparams: Namespace) -> BaseAggregator:
    """Creates aggregator object."""
    aggregators = {
        'SimpleAverageAggregator': (
            lambda hp: SimpleAverageAggregator()
        ),
        'ExponentialAverageAggregator': (
            lambda hp: ExponentialAverageAggregator()
        ),

        'ConcatGRUAggregator': (
            lambda hp: ConcatGRUAggregator(
                edge_dim=hp.dims_edge,
                node_dim=hp.dims_node,
            )
        ),
        'GRUAggregator': (
            lambda hp: GRUAggregator(edge_dim=hp.dims_edge)
        ),
    }

    if name not in aggregators.keys():
        raise RuntimeError(f'No such aggregator: \"{name}\"')

    return aggregators[name](hparams)
