"""Power Grid base networks."""

from collections import OrderedDict
from typing import List, Union, Sequence, Dict

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.inference import InferenceBlock
from torch import nn as nn


class BaseNet(nn.Module):
    """Feed forward base network.

    :param obs_shapes: The observation shape.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: The number of hidden units.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)],
                 hidden_units: List[int]):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.perception_dict = OrderedDict()

        # concatenate normalized and ready features
        in_shapes = [self.obs_shapes['features_normalize']] + [self.obs_shapes['features_ready']]
        self.perception_dict['all_features'] = ConcatenationBlock(
            in_keys=['features_normalize', 'features_ready'],
            in_shapes=in_shapes, out_keys='all_features', concat_dim=-1
        )

        # process features with vanilla dense net
        self.perception_dict['hidden_out'] = DenseBlock(
            in_keys='all_features', in_shapes=self.perception_dict['all_features'].out_shapes(),
            out_keys='hidden_out', hidden_units=hidden_units, non_lin=non_lin)

    def build_inference_block(self, out_keys: Union[str, List[str]]) -> InferenceBlock:
        """implementation of :class:`~maze.perception.blocks.inference.InferenceBlockBuilder` interface
        """
        in_keys = set(sum([block.in_keys for block in self.perception_dict.values()], []))
        in_keys = list(filter(lambda key: key in self.obs_shapes.keys(), in_keys))
        inference_block = InferenceBlock(
            in_keys=in_keys, out_keys=out_keys,
            in_shapes=[self.obs_shapes[key] for key in in_keys],
            perception_blocks=self.perception_dict)

        return inference_block
